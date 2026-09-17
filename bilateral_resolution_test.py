"""
Tests bilateral candidate resolution: generate LEFT and RIGHT
candidates independently, then jointly select the pair with the
best combined curvature score, subject only to the constraint that
the two chosen points are not the same physical contour location.

This does NOT require knowing landmark identity in advance — only
that two candidates aren't literally the same point — avoiding the
circularity flagged in the checkpoint.

Run: python bilateral_resolution_test.py
"""

import io
import requests
import numpy as np
import cv2
from PIL import Image

ALPHA_THRESHOLD = 10
MIN_DEFECT_DEPTH = 1000
CURVATURE_WINDOW = 8
NET_Y_WINDOW_PX = 300
MIN_ARC_BEFORE_PEAK_SEARCH = 10
MIN_PEAK_SEPARATION = 10
TOP_K = 8
SAME_POINT_RADIUS_PX = 15  # two candidates within this = same physical location

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"

TEST_CASES = [
    {"item_id": "ac7fffb7-2e3a-40f8-894a-0e85fc245373", "label": "polo",
     "confirmed_cuff": {"LEFT": (31, 154), "RIGHT": (439, 142)}},
    {"item_id": "25989827-db9d-4d65-90b3-12f5357b0b44", "label": "hanging_shirt",
     "confirmed_cuff": {"LEFT": (26, 457), "RIGHT": (442, 447)}},
    {"item_id": "2ea4bb57-a358-4349-8aaf-204ded0772ab", "label": "rugby",
     "confirmed_cuff": {"LEFT": (51, 488), "RIGHT": (430, 459)}},
    {"item_id": "6a53c00f-4f84-4e64-ad3b-5d2753504481", "label": "dress_shirt",
     "confirmed_cuff": {"LEFT": (54, 633), "RIGHT": (166, 660)},
     "out_of_scope_sides": ["RIGHT"]},
]


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255


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


def get_side_candidates(main_contour, underarm_idx):
    forward, backward = contour_arc_distances(main_contour, underarm_idx)
    net_y_fwd = get_net_y_change(main_contour, forward, NET_Y_WINDOW_PX)
    net_y_bwd = get_net_y_change(main_contour, backward, NET_Y_WINDOW_PX)
    sleeve_arc = forward if abs(net_y_fwd) < abs(net_y_bwd) else backward
    ordered = sorted(sleeve_arc.items(), key=lambda kv: kv[1])
    curvatures = compute_curvature(main_contour, ordered)
    return find_top_k_peaks(ordered, curvatures, TOP_K)


def main():
    correct_sides = 0
    total_sides = 0

    for case in TEST_CASES:
        out_of_scope = case.get("out_of_scope_sides", [])
        print(f"\n{'='*75}")
        print(f"{case['item_id']} — {case['label']}")
        print('='*75)

        result_data = requests.get(
            f"http://127.0.0.1:8000/api/wardrobe/{case['item_id']}?user_id={AVATAR_USER_ID}"
        ).json()["data"]
        image_bytes = requests.get(result_data["clean_image_url"]).content

        binary_mask = get_binary_mask(image_bytes)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_contour = max(contours, key=cv2.contourArea)
        w_img = binary_mask.shape[1]

        left_idx, right_idx = find_underarm_indices(main_contour, w_img)

        if "RIGHT" in out_of_scope:
            # only LEFT is in-scope; just report it independently
            left_peaks = get_side_candidates(main_contour, left_idx)
            c, d, idx = left_peaks[0]
            pt = tuple(int(v) for v in main_contour[idx][0])
            confirmed = case["confirmed_cuff"]["LEFT"]
            dist = np.hypot(pt[0]-confirmed[0], pt[1]-confirmed[1])
            match = dist < 15
            total_sides += 1
            if match: correct_sides += 1
            print(f"  LEFT only (RIGHT out of scope): chosen={pt}  confirmed={confirmed}  "
                  f"dist={dist:.1f}px  {'✅' if match else '❌'}")
            continue

        left_peaks = get_side_candidates(main_contour, left_idx)
        right_peaks = get_side_candidates(main_contour, right_idx)

        # BILATERAL RESOLUTION: try every valid (LEFT, RIGHT) pair,
        # score by combined curvature, require distinct physical points
        best_pair = None
        best_score = -1
        for lc, ld, lidx in left_peaks:
            lpt = tuple(int(v) for v in main_contour[lidx][0])
            for rc, rd, ridx in right_peaks:
                rpt = tuple(int(v) for v in main_contour[ridx][0])
                if np.hypot(lpt[0]-rpt[0], lpt[1]-rpt[1]) < SAME_POINT_RADIUS_PX:
                    continue  # same physical point, invalid pair
                score = lc + rc
                if score > best_score:
                    best_score = score
                    best_pair = (lpt, rpt, lc, rc)

        lpt, rpt, lc, rc = best_pair
        left_confirmed = case["confirmed_cuff"]["LEFT"]
        right_confirmed = case["confirmed_cuff"]["RIGHT"]
        left_dist = np.hypot(lpt[0]-left_confirmed[0], lpt[1]-left_confirmed[1])
        right_dist = np.hypot(rpt[0]-right_confirmed[0], rpt[1]-right_confirmed[1])
        left_match = left_dist < 15
        right_match = right_dist < 15

        total_sides += 2
        if left_match: correct_sides += 1
        if right_match: correct_sides += 1

        print(f"  Best bilateral pair (combined curvature={best_score:.1f}):")
        print(f"    LEFT:  chosen={lpt} (curv={lc:.1f}°)  confirmed={left_confirmed}  "
              f"dist={left_dist:.1f}px  {'✅' if left_match else '❌'}")
        print(f"    RIGHT: chosen={rpt} (curv={rc:.1f}°)  confirmed={right_confirmed}  "
              f"dist={right_dist:.1f}px  {'✅' if right_match else '❌'}")

    print(f"\n\n{'='*75}")
    print(f"FINAL: {correct_sides}/{total_sides}")
    print('='*75)


if __name__ == "__main__":
    main()