"""
Held-out validation for FROZEN V7 cuff detection.

Zero changes to V7's logic or constants. Runs against every real
wardrobe item NOT used to develop the algorithm (excludes only the
hanging-sleeve shirt and polo_baseline, which produced the
hypothesis itself).

Run: python cuff_v7_holdout_validation.py
"""

import io
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw

ALPHA_THRESHOLD = 10
MIN_DEFECT_DEPTH = 1000
CURVATURE_WINDOW = 8
CLASSIFICATION_WINDOW_PX = 60
MIN_ARC_BEFORE_PEAK_SEARCH = 30

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"

# Excluded — these two produced the V7 hypothesis, not held-out data
DEVELOPMENT_ITEM_IDS = [
    "25989827-db9d-4d65-90b3-12f5357b0b44",  # hanging_sleeve_shirt
    "ac7fffb7-2e3a-40f8-894a-0e85fc245373",  # polo_baseline
]


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255, image


def find_underarm_indices(main_contour, w_img):
    hull_indices = cv2.convexHull(main_contour, returnPoints=False)
    try:
        defects = cv2.convexityDefects(main_contour, hull_indices)
    except cv2.error:
        return None, None
    if defects is None:
        return None, None
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
    if not left_candidates or not right_candidates:
        return None, None
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
    result = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/?user_id={AVATAR_USER_ID}"
    ).json()["data"]

    all_items = result.get("items", [])
    holdout_items = [
        item for item in all_items
        if item["item_id"] not in DEVELOPMENT_ITEM_IDS
        and item.get("category") == "top"
        and not item.get("is_archived", False)
    ]

    print(f"Found {len(holdout_items)} held-out items (excluding the 2 V7 development garments)\n")

    for item in holdout_items:
        print(f"\n{'='*70}")
        print(f"{item['item_id']} — {item.get('item_type', 'unknown')}")
        print('='*70)

        image_bytes = requests.get(item["clean_image_url"]).content
        binary_mask, original = get_binary_mask(image_bytes)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_contour = max(contours, key=cv2.contourArea)
        w_img = binary_mask.shape[1]

        left_idx, right_idx = find_underarm_indices(main_contour, w_img)
        if left_idx is None:
            print("  Could not find both underarm points — skipping")
            continue

        canvas = original.convert("RGB").copy()
        draw = ImageDraw.Draw(canvas)
        full_pts = [(int(p[0][0]), int(p[0][1])) for p in main_contour]
        draw.line(full_pts + [full_pts[0]], fill=(60, 60, 60), width=1)

        for side_name, underarm_idx, going_positive in [
            ("LEFT", left_idx, False), ("RIGHT", right_idx, True)
        ]:
            underarm_x = int(main_contour[underarm_idx][0][0])
            forward, backward = contour_arc_distances(main_contour, underarm_idx)

            fwd_disp = classify_direction(main_contour, underarm_x, forward, CLASSIFICATION_WINDOW_PX)
            bwd_disp = classify_direction(main_contour, underarm_x, backward, CLASSIFICATION_WINDOW_PX)
            sleeve_arc = forward if fwd_disp > bwd_disp else backward

            window_end = find_x_return_boundary(main_contour, sleeve_arc, underarm_x, going_positive)
            stop_idx, stop_d, stop_c = find_max_curvature_in_window(main_contour, sleeve_arc, window_end)

            if stop_idx is None:
                print(f"  {side_name}: no cuff candidate found")
                continue

            stop_point = tuple(int(v) for v in main_contour[stop_idx][0])
            print(f"  {side_name}: boundary={window_end:.1f}px  cuff={stop_point}  curvature={stop_c:.1f}°")

            x, y = main_contour[underarm_idx][0]
            draw.ellipse([x-8, y-8, x+8, y+8], outline=(0, 255, 0), width=3)
            sx, sy = stop_point
            draw.ellipse([sx-8, sy-8, sx+8, sy+8], outline=(255, 255, 0), width=3)

        filename = f"holdout_cuff_{item['item_id'][:8]}.png"
        canvas.save(filename)
        print(f"  Saved: {filename}")


if __name__ == "__main__":
    main()