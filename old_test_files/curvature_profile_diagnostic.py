"""
Per-point curvature profile diagnostic.

Tests a genuinely different signal than every prior attempt: not
a single aggregate statistic on PATH_A/PATH_B, but the LOCAL
curvature at every point along each path. Hypothesis: a torso
boundary is smooth (low, consistent curvature); a sleeve boundary
narrows toward a cuff, producing sharper local bends (higher
curvature, likely concentrated near the sleeve's far end).

Outputs the full curvature sequence for visual/numeric inspection —
does NOT pre-decide what "sleeve-like" curvature looks like.

Run: python curvature_profile_diagnostic.py
"""

import sys
import io
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw

sys.path.insert(0, ".")
from core.storage.supabase_client import supabase

ALPHA_THRESHOLD = 10
MIN_DEFECT_DEPTH = 1000
CURVATURE_WINDOW = 8  # points on each side used to estimate local 
                        # curvature — smooths pixel-level noise

TEST_CASES = [
    {"item_id": "25989827-db9d-4d65-90b3-12f5357b0b44", "label": "hanger_shirt"},
    {"item_id": "ba2e5aa1-bfb5-462f-aaad-fe378f658c42", "label": "long_sleeve_hem_failure"},
    {"item_id": "ac7fffb7-2e3a-40f8-894a-0e85fc245373", "label": "polo_baseline"},
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
    left_idx = left_candidates[0][1] if left_candidates else None
    right_idx = right_candidates[0][1] if right_candidates else None
    return left_idx, right_idx


def split_contour_paths(main_contour, left_idx, right_idx):
    lo, hi = sorted([left_idx, right_idx])
    path_a = main_contour[lo:hi+1]
    path_b = np.concatenate([main_contour[hi:], main_contour[:lo+1]])
    return path_a, path_b


def compute_curvature_profile(path):
    """
    At each point, measures the angle change between the incoming
    and outgoing direction vectors (using CURVATURE_WINDOW points
    of lookback/lookahead to smooth pixel noise). Returns one
    curvature value per point — a smooth boundary gives small,
    consistent values; a sharp bend gives a spike.
    """
    pts = path[:, 0, :].astype(float)
    n = len(pts)
    curvatures = []

    for i in range(n):
        prev_i = max(0, i - CURVATURE_WINDOW)
        next_i = min(n - 1, i + CURVATURE_WINDOW)
        if next_i - prev_i < 2:
            curvatures.append(0.0)
            continue

        v1 = pts[i] - pts[prev_i]
        v2 = pts[next_i] - pts[i]

        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            curvatures.append(0.0)
            continue

        cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        curvatures.append(angle)

    return curvatures


def visualize_curvature(image, path, curvatures, w_img, h_img, max_curve):
    """Colors each point along the path by its curvature — 
    blue (smooth) to red (sharp bend)."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    for i, pt in enumerate(path):
        x, y = int(pt[0][0]), int(pt[0][1])
        intensity = min(curvatures[i] / max_curve, 1.0) if max_curve > 0 else 0
        color = (int(255 * intensity), 0, int(255 * (1 - intensity)))
        draw.ellipse([x-2, y-2, x+2, y+2], fill=color)

    return canvas


def main():
    for case in TEST_CASES:
        print(f"\n{'='*60}")
        print(f"{case['item_id']} — {case['label']}")
        print('='*60)

        result = supabase.table("closet_items")\
            .select("clean_image_url").eq("item_id", case["item_id"]).single().execute()
        image_bytes = requests.get(result.data["clean_image_url"]).content

        binary_mask, original = get_binary_mask(image_bytes)
        h_img, w_img = binary_mask.shape

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_contour = max(contours, key=cv2.contourArea)

        left_idx, right_idx = find_underarm_indices(main_contour, w_img)
        if left_idx is None or right_idx is None:
            print("  Could not find both underarm points — skipping")
            continue

        path_a, path_b = split_contour_paths(main_contour, left_idx, right_idx)

        curv_a = compute_curvature_profile(path_a)
        curv_b = compute_curvature_profile(path_b)

        print(f"  PATH_A — mean curvature: {np.mean(curv_a):.1f}°, "
              f"max: {np.max(curv_a):.1f}°, std: {np.std(curv_a):.1f}°")
        print(f"  PATH_B — mean curvature: {np.mean(curv_b):.1f}°, "
              f"max: {np.max(curv_b):.1f}°, std: {np.std(curv_b):.1f}°")

        all_max = max(np.max(curv_a), np.max(curv_b))

        annotated = original.convert("RGB").copy()
        annotated = visualize_curvature(annotated, path_a, curv_a, w_img, h_img, all_max)
        annotated = visualize_curvature(annotated, path_b, curv_b, w_img, h_img, all_max)
        filename = f"curvature_{case['label']}.png"
        annotated.save(filename)
        print(f"  Saved: {filename} (blue=smooth, red=sharp bend)")


if __name__ == "__main__":
    main()