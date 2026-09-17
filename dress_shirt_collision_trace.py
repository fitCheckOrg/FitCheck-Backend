"""
Full diagnostic trace for the dress-shirt (6a53c00f) LEFT/RIGHT
collision. Prints every intermediate value specified — no changes
to V7's logic, purely instrumentation.

Run: python dress_shirt_collision_trace.py
"""

import io
import requests
import numpy as np
import cv2
from PIL import Image

ALPHA_THRESHOLD = 10
MIN_DEFECT_DEPTH = 1000
CURVATURE_WINDOW = 8
CLASSIFICATION_WINDOW_PX = 60
MIN_ARC_BEFORE_PEAK_SEARCH = 30

ITEM_ID = "6a53c00f-4f84-4e64-ad3b-5d2753504481"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"


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


def classify_direction(contour, underarm_x, arc_dict, window_px):
    max_disp = 0
    for idx, d in arc_dict.items():
        if d > window_px:
            continue
        x = contour[idx][0][0]
        max_disp = max(max_disp, abs(int(x) - underarm_x))
    return max_disp


def find_x_return_boundary(contour, sleeve_arc, underarm_x, going_positive):
    ordered = sorted(sleeve_arc.items(), key=lambda kv: kv[1])
    moved_away = False
    for idx, d in ordered:
        if d < MIN_ARC_BEFORE_PEAK_SEARCH:
            continue
        x = int(contour[idx][0][0])
        if going_positive:
            if x > underarm_x:
                moved_away = True
            elif moved_away and x <= underarm_x:
                return d, idx
        else:
            if x < underarm_x:
                moved_away = True
            elif moved_away and x >= underarm_x:
                return d, idx
    last_idx, last_d = ordered[-1]
    return last_d, last_idx


def compute_curvature_at_indices(contour, arc_dict, window=CURVATURE_WINDOW):
    ordered = sorted(arc_dict.items(), key=lambda kv: kv[1])
    indices = [idx for idx, d in ordered]
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
        if norm1 == 0 or norm2 == 0:
            curvatures[indices[i]] = 0.0
            continue
        cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        curvatures[indices[i]] = np.degrees(np.arccos(cos_angle))
    return curvatures


def find_max_curvature_in_window(contour, sleeve_arc, window_end):
    curvatures = compute_curvature_at_indices(contour, sleeve_arc)
    best_idx, best_d, best_c = None, None, -1
    for idx, d in sleeve_arc.items():
        if d < MIN_ARC_BEFORE_PEAK_SEARCH or d > window_end:
            continue
        c = curvatures.get(idx, 0.0)
        if c > best_c:
            best_c, best_idx, best_d = c, idx, d
    return best_idx, best_d, best_c


def main():
    result_data = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{ITEM_ID}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    image_bytes = requests.get(result_data["clean_image_url"]).content

    binary_mask = get_binary_mask(image_bytes)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)
    w_img = binary_mask.shape[1]
    print(f"Contour has {len(main_contour)} points, image width={w_img}")

    left_idx, right_idx = find_underarm_indices(main_contour, w_img)

    for side_name, underarm_idx, going_positive in [
        ("LEFT", left_idx, False), ("RIGHT", right_idx, True)
    ]:
        underarm_x = int(main_contour[underarm_idx][0][0])
        underarm_pt = tuple(int(v) for v in main_contour[underarm_idx][0])
        forward, backward = contour_arc_distances(main_contour, underarm_idx)

        fwd_disp = classify_direction(main_contour, underarm_x, forward, CLASSIFICATION_WINDOW_PX)
        bwd_disp = classify_direction(main_contour, underarm_x, backward, CLASSIFICATION_WINDOW_PX)
        chosen_direction = "FORWARD" if fwd_disp > bwd_disp else "BACKWARD"
        sleeve_arc = forward if fwd_disp > bwd_disp else backward

        boundary_d, boundary_idx = find_x_return_boundary(main_contour, sleeve_arc, underarm_x, going_positive)
        boundary_pt = tuple(int(v) for v in main_contour[boundary_idx][0])

        stop_idx, stop_d, stop_c = find_max_curvature_in_window(main_contour, sleeve_arc, boundary_d)
        stop_pt = tuple(int(v) for v in main_contour[stop_idx][0]) if stop_idx else None

        print(f"\n{'='*60}")
        print(f"{side_name}")
        print(f"{'='*60}")
        print(f"  underarm_index: {underarm_idx}")
        print(f"  underarm_coordinate: {underarm_pt}")
        print(f"  going_positive: {going_positive}")
        print(f"  forward_displacement: {fwd_disp}")
        print(f"  backward_displacement: {bwd_disp}")
        print(f"  chosen_direction: {chosen_direction}")
        print(f"  underarm_x: {underarm_x}")
        print(f"  x_return_boundary_arc: {boundary_d:.1f}")
        print(f"  boundary_index: {boundary_idx}")
        print(f"  boundary_coordinate: {boundary_pt}")
        print(f"  max_curvature_index: {stop_idx}")
        print(f"  max_curvature_coordinate: {stop_pt}")
        print(f"  max_curvature_value: {stop_c:.1f}")


if __name__ == "__main__":
    main()