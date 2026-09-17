"""
Visual check for both held-out items, plus a direct trace of
whether ba2e5aa1's apparent LEFT/RIGHT swap is a real geometric bug
or a labeling mismatch — by printing raw underarm assignment before
any resolution logic runs.

Run: python holdout_visual_and_trace.py
"""

import io
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw

ALPHA_THRESHOLD = 10
MIN_DEFECT_DEPTH = 1000
CURVATURE_WINDOW = 8
NET_Y_WINDOW_PX = 300
MIN_ARC_BEFORE_PEAK_SEARCH = 10
MIN_PEAK_SEPARATION = 10
TOP_K = 8
COLLAR_ZONE_Y_PCT = 15
POINT_EXCLUSION_RADIUS_PX = 15
COLLISION_RADIUS_PX = 15
REGION_EXCLUSION_RADIUS_PX = 100

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"

TEST_CASES = [
    {"item_id": "e398e450-a4be-447a-b981-5a57297c7e66", "label": "turtleneck"},
    {"item_id": "ba2e5aa1-bfb5-462f-aaad-fe378f658c42", "label": "long_sleeve_shirt"},
]


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255, image


def find_underarm_indices(main_contour, w_img):
    hull_indices = cv2.convexHull(main_contour, returnPoints=False)
    defects = cv2.convexityDefects(main_contour, hull_indices)
    defects_flat = defects.reshape(-1, 4)
    left_candidates, right_candidates = [], []
    for i in range(len(defects_flat)):
        _, _, far_idx, depth = defects_flat[i]
        depth, far_idx = int(depth), int(far_idx)
        if depth <= MIN_DEFECT_DEPTH:
            continue
        far_pt = main_contour[far_idx][0]
        x_pct = far_pt[0] / w_img * 100
        (left_candidates if x_pct < 50 else right_candidates).append((depth, far_idx))
    left_candidates.sort(reverse=True)
    right_candidates.sort(reverse=True)
    return left_candidates[0][1], right_candidates[0][1]


def contour_arc_distances(contour, start_idx):
    n = len(contour)
    forward, backward = {}, {}
    dist = 0.0
    for step in range(n):
        idx = (start_idx + step) % n
        forward[idx] = dist
        next_idx = (idx + 1) % n
        dist += np.linalg.norm(contour[next_idx][0].astype(float) - contour[idx][0].astype(float))
    dist = 0.0
    for step in range(n):
        idx = (start_idx - step) % n
        backward[idx] = dist
        prev_idx = (idx - 1) % n
        dist += np.linalg.norm(contour[prev_idx][0].astype(float) - contour[idx][0].astype(float))
    return forward, backward


def get_net_y_change(contour, arc_dict, window_px):
    ordered = sorted(arc_dict.items(), key=lambda kv: kv[1])
    points = [(idx, d) for idx, d in ordered if d <= window_px]
    if len(points) < 2:
        return None
    y_first = int(contour[points[0][0]][0][1])
    y_last = int(contour[points[-1][0]][0][1])
    return y_last - y_first


def compute_curvature(contour, ordered_arc, window=CURVATURE_WINDOW):
    indices = [idx for idx, d in ordered_arc]
    pts = np.array([contour[i][0] for i in indices], dtype=float)
    n = len(pts)
    curvatures = {}
    for i in range(n):
        prev_i = max(0, i - window)
        next_i = min(n - 1, i + window)
        if next_i - prev_i < 2:
            curvatures[indices[i]] = 0.0
            continue
        v1 = pts[i] - pts[prev_i]
        v2 = pts[next_i] - pts[i]
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        curvatures[indices[i]] = 0.0 if norm1 == 0 or norm2 == 0 else \
            np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)))
    return curvatures


def find_top_k_peaks(ordered_arc, curvatures, top_k, min_separation=MIN_PEAK_SEPARATION):
    candidates = []
    for i in range(1, len(ordered_arc) - 1):
        idx, d = ordered_arc[i]
        if d < MIN_ARC_BEFORE_PEAK_SEARCH:
            continue
        prev_idx, _ = ordered_arc[i - 1]
        next_idx, _ = ordered_arc[i + 1]
        c = curvatures.get(idx, 0.0)
        if c > curvatures.get(prev_idx, 0.0) and c > curvatures.get(next_idx, 0.0):
            candidates.append((c, d, idx))
    candidates.sort(reverse=True)
    selected = []
    for c, d, idx in candidates:
        if not any(abs(d - sd) < min_separation for _, sd, _ in selected):
            selected.append((c, d, idx))
        if len(selected) >= top_k:
            break
    return selected


def get_filtered_candidates(main_contour, underarm_idx, h_img, point_landmarks):
    forward, backward = contour_arc_distances(main_contour, underarm_idx)
    net_y_fwd = get_net_y_change(main_contour, forward, NET_Y_WINDOW_PX)
    net_y_bwd = get_net_y_change(main_contour, backward, NET_Y_WINDOW_PX)
    sleeve_arc = forward if abs(net_y_fwd) < abs(net_y_bwd) else backward
    ordered = sorted(sleeve_arc.items(), key=lambda kv: kv[1])
    curvatures = compute_curvature(main_contour, ordered)
    peaks = find_top_k_peaks(ordered, curvatures, TOP_K)
    filtered = []
    for c, d, idx in peaks:
        pt = tuple(int(v) for v in main_contour[idx][0])
        y_pct = pt[1] / h_img * 100
        is_known = any(np.hypot(pt[0]-lm[0], pt[1]-lm[1]) < POINT_EXCLUSION_RADIUS_PX
                        for lm in point_landmarks)
        is_collar = y_pct < COLLAR_ZONE_Y_PCT
        if not is_known and not is_collar:
            filtered.append((c, d, pt))
    return filtered


def resolve_combined(left_candidates, right_candidates):
    li, ri = 0, 0
    excluded_regions_for_left = []
    excluded_regions_for_right = []
    while li < len(left_candidates) and ri < len(right_candidates):
        lc, ld, lpt = left_candidates[li]
        rc, rd, rpt = right_candidates[ri]
        if any(np.hypot(lpt[0]-r[0], lpt[1]-r[1]) < REGION_EXCLUSION_RADIUS_PX for r in excluded_regions_for_left):
            li += 1
            continue
        if any(np.hypot(rpt[0]-r[0], rpt[1]-r[1]) < REGION_EXCLUSION_RADIUS_PX for r in excluded_regions_for_right):
            ri += 1
            continue
        same_point_collision = np.hypot(lpt[0]-rpt[0], lpt[1]-rpt[1]) < COLLISION_RADIUS_PX
        if not same_point_collision:
            return (lpt, lc, ld), (rpt, rc, rd)
        if ld <= rd:
            excluded_regions_for_right.append(rpt)
            ri += 1
        else:
            excluded_regions_for_left.append(lpt)
            li += 1
    left_result = left_candidates[li] if li < len(left_candidates) else None
    right_result = right_candidates[ri] if ri < len(right_candidates) else None
    return (
        (left_result[2], left_result[0], left_result[1]) if left_result else None,
        (right_result[2], right_result[0], right_result[1]) if right_result else None
    )


def main():
    for case in TEST_CASES:
        print(f"\n{'='*75}")
        print(f"{case['item_id']} — {case['label']}")
        print('='*75)

        result_data = requests.get(
            f"http://127.0.0.1:8000/api/wardrobe/{case['item_id']}?user_id={AVATAR_USER_ID}"
        ).json()["data"]
        image_bytes = requests.get(result_data["clean_image_url"]).content

        binary_mask, original = get_binary_mask(image_bytes)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_contour = max(contours, key=cv2.contourArea)
        h_img, w_img = binary_mask.shape

        left_idx, right_idx = find_underarm_indices(main_contour, w_img)
        left_underarm_pt = tuple(int(v) for v in main_contour[left_idx][0])
        right_underarm_pt = tuple(int(v) for v in main_contour[right_idx][0])

        print(f"RAW underarm assignment: LEFT_underarm={left_underarm_pt}  RIGHT_underarm={right_underarm_pt}")

        underarms = [left_underarm_pt, right_underarm_pt]
        left_c = get_filtered_candidates(main_contour, left_idx, h_img, underarms)
        right_c = get_filtered_candidates(main_contour, right_idx, h_img, underarms)

        left_result, right_result = resolve_combined(left_c, right_c)

        canvas = original.convert("RGB").copy()
        draw = ImageDraw.Draw(canvas)
        for pt, color, label in [(left_underarm_pt, (0,255,0), "L_underarm"),
                                    (right_underarm_pt, (0,255,0), "R_underarm")]:
            x, y = pt
            draw.ellipse([x-6,y-6,x+6,y+6], outline=color, width=2)
            draw.text((x+8,y-6), label, fill=color)

        for side_name, result, color in [("LEFT", left_result, (255,0,0)), ("RIGHT", right_result, (0,150,255))]:
            if result:
                pt, c, d = result
                print(f"  {side_name} chosen: {pt}  curvature={c:.1f}°  arc={d:.1f}px")
                x, y = pt
                draw.ellipse([x-8,y-8,x+8,y+8], outline=color, width=3)
                draw.text((x+10,y-8), f"{side_name} cuff", fill=color)

        filename = f"holdout_{case['label']}.png"
        canvas.save(filename)
        print(f"Saved: {filename}")


if __name__ == "__main__":
    main()