"""
Tests: stop the search when arc gets closest to the OTHER side's
underarm coordinate — a real landmark, not a distance guess.

Separately reports whether MIN_ARC_BEFORE_PEAK_SEARCH still causes
a problem on dress_shirt LEFT even with this new boundary, since
hand-tracing suggested this boundary alone may not fix that case.

Run: python other_underarm_boundary_test.py
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
MIN_ARC_BEFORE_PEAK_SEARCH = 30

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


def find_closest_approach_to_point(contour, arc_dict, target_point):
    tx, ty = target_point
    best_d, best_dist = None, float('inf')
    for idx, d in arc_dict.items():
        x, y = contour[idx][0]
        dist = np.hypot(int(x) - tx, int(y) - ty)
        if dist < best_dist:
            best_dist, best_d = dist, d
    return best_d, best_dist


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


def find_top_peak(contour, arc_dict, min_arc, max_arc):
    ordered = [(idx, d) for idx, d in sorted(arc_dict.items(), key=lambda kv: kv[1])
               if min_arc <= d <= max_arc]
    if len(ordered) < 3:
        return None, None, None
    curvatures = compute_curvature(contour, ordered)
    best_idx, best_d, best_c = None, None, -1
    for idx, d in ordered:
        c = curvatures.get(idx, 0.0)
        if c > best_c:
            best_c, best_idx, best_d = c, idx, d
    return best_idx, best_d, best_c


def main():
    correct = 0
    total = 0

    for case in TEST_CASES:
        print(f"\n{'='*70}")
        print(f"{case['item_id']} — {case['label']}")
        print('='*70)

        result_data = requests.get(
            f"http://127.0.0.1:8000/api/wardrobe/{case['item_id']}?user_id={AVATAR_USER_ID}"
        ).json()["data"]
        image_bytes = requests.get(result_data["clean_image_url"]).content

        binary_mask = get_binary_mask(image_bytes)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_contour = max(contours, key=cv2.contourArea)
        w_img = binary_mask.shape[1]

        left_idx, right_idx = find_underarm_indices(main_contour, w_img)
        underarms = {"LEFT": tuple(int(v) for v in main_contour[left_idx][0]),
                     "RIGHT": tuple(int(v) for v in main_contour[right_idx][0])}

        for side_name, underarm_idx in [("LEFT", left_idx), ("RIGHT", right_idx)]:
            other_side = "RIGHT" if side_name == "LEFT" else "LEFT"
            other_underarm = underarms[other_side]

            forward, backward = contour_arc_distances(main_contour, underarm_idx)
            net_y_fwd = get_net_y_change(main_contour, forward, NET_Y_WINDOW_PX)
            net_y_bwd = get_net_y_change(main_contour, backward, NET_Y_WINDOW_PX)
            sleeve_arc = forward if abs(net_y_fwd) < abs(net_y_bwd) else backward

            boundary_arc, boundary_dist = find_closest_approach_to_point(main_contour, sleeve_arc, other_underarm)

            stop_idx, stop_d, stop_c = find_top_peak(main_contour, sleeve_arc, MIN_ARC_BEFORE_PEAK_SEARCH, boundary_arc)

            confirmed = case["confirmed_cuff"][side_name]
            found_pt = tuple(int(v) for v in main_contour[stop_idx][0]) if stop_idx else None
            dist = np.hypot(found_pt[0] - confirmed[0], found_pt[1] - confirmed[1]) if found_pt else None

            total += 1
            match = dist is not None and dist < 15
            if match:
                correct += 1

            print(f"  {side_name}: boundary=other_underarm_closest_approach@{boundary_arc:.0f}px "
                  f"(dist={boundary_dist:.0f}px)  found={found_pt} (curv={stop_c if stop_c else 0:.1f}°)  "
                  f"confirmed={confirmed}  dist={dist if dist else -1:.1f}px  {'✅' if match else '❌'}")

    print(f"\n\n{'='*70}")
    print(f"FINAL: {correct}/{total}")
    print('='*70)


if __name__ == "__main__":
    main()