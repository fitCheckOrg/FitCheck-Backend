"""
Adaptive cuff trace — pure diagnostic, no classification rule.

Uses the already-validated Layer 1 rule (min |net_y_change|) to
select sleeve direction. Then traces the ENTIRE selected arc
(unbounded by any fixed window), recording arc distance, x, y,
dx, dy, curvature, and displacement-from-underarm at regular
intervals. Separately finds ALL local curvature peaks across the
full arc, and reports each peak's distance from the CONFIRMED cuff
position (known ground truth), so we can see directly whether a
consistent geometric signature exists near the true cuff across
all four garments — without deciding X vs Y orientation first.

Run: python adaptive_cuff_trace.py
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
TRACE_CAP = 900
PRINT_INTERVAL = 25

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"

TEST_CASES = [
    {"item_id": "ac7fffb7-2e3a-40f8-894a-0e85fc245373", "label": "polo",
     "confirmed_cuff": {"LEFT": (31, 154), "RIGHT": (439, 142)}},
    {"item_id": "25989827-db9d-4d65-90b3-12f5357b0b44", "label": "hanging_shirt",
     "confirmed_cuff": {"LEFT": (26, 457), "RIGHT": (442, 447)}},
    {"item_id": "2ea4bb57-a358-4349-8aaf-204ded0772ab", "label": "rugby",
     "confirmed_cuff": {"LEFT": (51, 488), "RIGHT": (430, 459)}},
    {"item_id": "6a53c00f-4f84-4e64-ad3b-5d2753504481", "label": "dress_shirt",
     "confirmed_cuff": {"LEFT": (54, 633), "RIGHT": (166, 660)}},
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


def compute_curvature_and_deltas(contour, ordered_arc, window=CURVATURE_WINDOW):
    """Returns dict idx -> (dx, dy, curvature), using the same
    window as every validated curvature computation tonight."""
    indices = [idx for idx, d in ordered_arc]
    pts = np.array([contour[i][0] for i in indices], dtype=float)
    n = len(pts)
    results = {}
    for i in range(n):
        prev_i = max(0, i - window)
        next_i = min(n - 1, i + window)
        if next_i - prev_i < 2:
            results[indices[i]] = (0.0, 0.0, 0.0)
            continue
        v1 = pts[i] - pts[prev_i]
        v2 = pts[next_i] - pts[i]
        dx = pts[next_i][0] - pts[prev_i][0]
        dy = pts[next_i][1] - pts[prev_i][1]
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            curvature = 0.0
        else:
            cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
            curvature = np.degrees(np.arccos(cos_angle))
        results[indices[i]] = (dx, dy, curvature)
    return results


def find_all_local_peaks(ordered_arc, deltas, min_separation=10):
    candidates = []
    for i in range(1, len(ordered_arc) - 1):
        idx, d = ordered_arc[i]
        prev_idx, _ = ordered_arc[i - 1]
        next_idx, _ = ordered_arc[i + 1]
        c = deltas[idx][2]
        if c > deltas[prev_idx][2] and c > deltas[next_idx][2]:
            candidates.append((c, d, idx))
    candidates.sort(reverse=True)
    selected = []
    for c, d, idx in candidates:
        if not any(abs(d - sd) < min_separation for _, sd, _ in selected):
            selected.append((c, d, idx))
    return selected


def find_closest_arc_to_point(contour, arc_dict, target_point):
    tx, ty = target_point
    best_idx, best_d, best_dist = None, None, float('inf')
    for idx, d in arc_dict.items():
        x, y = contour[idx][0]
        dist = np.hypot(int(x) - tx, int(y) - ty)
        if dist < best_dist:
            best_dist, best_idx, best_d = dist, idx, d
    return best_d, best_dist


def main():
    for case in TEST_CASES:
        print(f"\n\n{'#'*80}")
        print(f"# {case['item_id']} — {case['label']}")
        print(f"{'#'*80}")

        result_data = requests.get(
            f"http://127.0.0.1:8000/api/wardrobe/{case['item_id']}?user_id={AVATAR_USER_ID}"
        ).json()["data"]
        image_bytes = requests.get(result_data["clean_image_url"]).content

        binary_mask = get_binary_mask(image_bytes)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_contour = max(contours, key=cv2.contourArea)
        w_img = binary_mask.shape[1]

        left_idx, right_idx = find_underarm_indices(main_contour, w_img)

        for side_name, underarm_idx in [("LEFT", left_idx), ("RIGHT", right_idx)]:
            underarm_x = int(main_contour[underarm_idx][0][0])
            underarm_y = int(main_contour[underarm_idx][0][1])
            forward, backward = contour_arc_distances(main_contour, underarm_idx)

            net_y_fwd = get_net_y_change(main_contour, forward, NET_Y_WINDOW_PX)
            net_y_bwd = get_net_y_change(main_contour, backward, NET_Y_WINDOW_PX)
            sleeve_arc = forward if abs(net_y_fwd) < abs(net_y_bwd) else backward
            direction_name = "FORWARD" if sleeve_arc is forward else "BACKWARD"

            confirmed_cuff = case["confirmed_cuff"][side_name]
            confirmed_arc, confirmed_dist = find_closest_arc_to_point(main_contour, sleeve_arc, confirmed_cuff)

            print(f"\n{'='*75}")
            print(f"{side_name} — underarm=({underarm_x},{underarm_y})  direction={direction_name}")
            print(f"CONFIRMED cuff={confirmed_cuff} is at arc={confirmed_arc:.1f}px "
                  f"(nearest contour point, {confirmed_dist:.1f}px away)")
            print(f"{'='*75}")

            ordered = [(idx, d) for idx, d in sorted(sleeve_arc.items(), key=lambda kv: kv[1]) if d <= TRACE_CAP]
            deltas = compute_curvature_and_deltas(main_contour, ordered)

            print(f"\n  --- Regular trace (every ~{PRINT_INTERVAL}px) ---")
            last_printed = -PRINT_INTERVAL
            for idx, d in ordered:
                if d - last_printed >= PRINT_INTERVAL:
                    x, y = main_contour[idx][0]
                    dx, dy, curv = deltas[idx]
                    x_disp = int(x) - underarm_x
                    y_disp = int(y) - underarm_y
                    marker = "  <<< NEAR CONFIRMED CUFF" if abs(d - confirmed_arc) < PRINT_INTERVAL else ""
                    print(f"    arc={d:6.1f}  pos=({int(x):3d},{int(y):3d})  "
                          f"dx={dx:+6.1f} dy={dy:+6.1f}  curvature={curv:6.1f}°  "
                          f"x_disp={x_disp:+4d} y_disp={y_disp:+4d}{marker}")
                    last_printed = d

            print(f"\n  --- All local curvature peaks (unbounded) ---")
            peaks = find_all_local_peaks(ordered, deltas)
            for rank, (c, d, idx) in enumerate(peaks[:8], 1):
                x, y = main_contour[idx][0]
                dist_from_confirmed = abs(d - confirmed_arc)
                flag = "  <<< CLOSEST TO CONFIRMED CUFF" if dist_from_confirmed == min(abs(pd - confirmed_arc) for _, pd, _ in peaks) else ""
                print(f"    rank={rank}  arc={d:6.1f}  pos=({int(x):3d},{int(y):3d})  "
                      f"curvature={c:6.1f}°  dist_from_confirmed_arc={dist_from_confirmed:6.1f}px{flag}")


if __name__ == "__main__":
    main()