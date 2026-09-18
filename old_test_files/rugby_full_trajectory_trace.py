"""
Full x,y trajectory trace across the rugby shirt's entire sleeve
arc, well beyond the x-return boundary, to determine whether:

(a) the boundary is cutting off BEFORE reaching the real cuff 
    region (a boundary-timing problem), or
(b) the arc genuinely reaches the cuff region but shows no strong 
    curvature there (a real signal-absence finding)

Prints the full path, unsampled resolution, with the x-return
boundary explicitly marked so we can see exactly where it falls
relative to the garment's actual lowest/farthest sleeve extent.

Run: python rugby_full_trajectory_trace.py
"""

import io
import requests
import numpy as np
import cv2
from PIL import Image

ALPHA_THRESHOLD = 10
MIN_DEFECT_DEPTH = 1000
CLASSIFICATION_WINDOW_PX = 60
MIN_ARC_BEFORE_PEAK_SEARCH = 30
PRINT_INTERVAL = 15
TRACE_CAP = 500  # well beyond the confirmed 255px boundary, to see 
                   # what's actually further along

ITEM_ID = "2ea4bb57-a358-4349-8aaf-204ded0772ab"
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
                return d
        else:
            if x < underarm_x:
                moved_away = True
            elif moved_away and x >= underarm_x:
                return d
    return ordered[-1][1]


def main():
    result_data = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{ITEM_ID}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    image_bytes = requests.get(result_data["clean_image_url"]).content

    binary_mask = get_binary_mask(image_bytes)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)
    w_img = binary_mask.shape[1]

    left_idx, right_idx = find_underarm_indices(main_contour, w_img)

    for side_name, underarm_idx, going_positive in [
        ("LEFT", left_idx, False), ("RIGHT", right_idx, True)
    ]:
        underarm_x = int(main_contour[underarm_idx][0][0])
        underarm_pt = tuple(int(v) for v in main_contour[underarm_idx][0])
        forward, backward = contour_arc_distances(main_contour, underarm_idx)

        fwd_disp = classify_direction(main_contour, underarm_x, forward, CLASSIFICATION_WINDOW_PX)
        bwd_disp = classify_direction(main_contour, underarm_x, backward, CLASSIFICATION_WINDOW_PX)
        sleeve_arc = forward if fwd_disp > bwd_disp else backward

        boundary = find_x_return_boundary(main_contour, sleeve_arc, underarm_x, going_positive)

        print(f"\n{'='*60}")
        print(f"{side_name} — underarm={underarm_pt}  CONFIRMED BOUNDARY={boundary:.1f}px")
        print(f"{'='*60}")

        ordered = sorted(sleeve_arc.items(), key=lambda kv: kv[1])
        last_printed = -PRINT_INTERVAL
        boundary_flagged = False

        max_y_seen = -1
        max_y_arc = None

        for idx, d in ordered:
            if d > TRACE_CAP:
                break
            x, y = main_contour[idx][0]
            if y > max_y_seen:
                max_y_seen = y
                max_y_arc = d

            if d - last_printed >= PRINT_INTERVAL:
                marker = ""
                if not boundary_flagged and d >= boundary:
                    marker = "  <<< X-RETURN BOUNDARY HERE"
                    boundary_flagged = True
                print(f"  arc={d:6.1f}px  pos=({int(x):3d},{int(y):3d}){marker}")
                last_printed = d

        print(f"\n  Deepest point (max y) reached in this trace: y={max_y_seen} at arc={max_y_arc:.1f}px")
        print(f"  (Confirmed boundary was {boundary:.1f}px — "
              f"{'boundary comes BEFORE deepest point' if boundary < max_y_arc else 'boundary comes AFTER deepest point'})")


if __name__ == "__main__":
    main()