"""
Path geometry diagnostic — neutral PATH_A/PATH_B, no upper/lower
labels. Tests the "central-X-coverage" hypothesis: does the torso
path spend more of its length near the garment's horizontal
centerline than the sleeve path does?

Run: python path_geometry_diagnostic.py
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
CENTER_ZONE_WIDTH_PCT = 30  # "central" = within this % of image 
                              # width from the horizontal center, 
                              # e.g. 30 means the middle 30% band

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
    """Neutral split — no upper/lower assignment, just A and B."""
    lo, hi = sorted([left_idx, right_idx])
    path_a = main_contour[lo:hi+1]
    path_b = np.concatenate([main_contour[hi:], main_contour[:lo+1]])
    return path_a, path_b


def analyze_path(path, w_img, h_img):
    xs = path[:, 0, 0]
    ys = path[:, 0, 1]

    center_x = w_img / 2
    zone_half_width = w_img * (CENTER_ZONE_WIDTH_PCT / 100) / 2
    in_center_zone = np.abs(xs - center_x) <= zone_half_width
    central_x_coverage_pct = round(np.sum(in_center_zone) / len(path) * 100, 1)

    lowest_idx_in_path = np.argmax(ys)
    lowest_point = (int(xs[lowest_idx_in_path]), int(ys[lowest_idx_in_path]))
    lowest_x_pct = round(lowest_point[0] / w_img * 100, 1)
    lowest_x_dist_from_center = round(abs(lowest_x_pct - 50), 1)

    # Arc length — sum of distances between consecutive points
    diffs = np.diff(path[:, 0, :], axis=0)
    arc_length = float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))

    return {
        "num_points": len(path),
        "arc_length": round(arc_length, 1),
        "min_x_pct": round(np.min(xs) / w_img * 100, 1),
        "max_x_pct": round(np.max(xs) / w_img * 100, 1),
        "min_y_pct": round(np.min(ys) / h_img * 100, 1),
        "max_y_pct": round(np.max(ys) / h_img * 100, 1),
        "lowest_point_x_pct": lowest_x_pct,
        "lowest_point_y_pct": round(lowest_point[1] / h_img * 100, 1),
        "lowest_point_dist_from_center_x": lowest_x_dist_from_center,
        "central_x_coverage_pct": central_x_coverage_pct,
    }


def visualize(image, path_a, path_b, left_idx, right_idx, main_contour):
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    pts_a = [(int(p[0][0]), int(p[0][1])) for p in path_a]
    pts_b = [(int(p[0][0]), int(p[0][1])) for p in path_b]
    if len(pts_a) > 1:
        draw.line(pts_a, fill=(0, 255, 255), width=3)  # PATH_A cyan
    if len(pts_b) > 1:
        draw.line(pts_b, fill=(255, 0, 255), width=3)  # PATH_B magenta

    left_pt = tuple(int(v) for v in main_contour[left_idx][0])
    right_pt = tuple(int(v) for v in main_contour[right_idx][0])
    for pt, color in [(left_pt, (255, 0, 0)), (right_pt, (0, 255, 0))]:
        x, y = pt
        draw.ellipse([x-8, y-8, x+8, y+8], fill=color)

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

        stats_a = analyze_path(path_a, w_img, h_img)
        stats_b = analyze_path(path_b, w_img, h_img)

        print("  PATH_A:")
        for k, v in stats_a.items():
            print(f"    {k}: {v}")
        print("  PATH_B:")
        for k, v in stats_b.items():
            print(f"    {k}: {v}")

        annotated = visualize(original, path_a, path_b, left_idx, right_idx, main_contour)
        filename = f"pathdiag_{case['label']}.png"
        annotated.save(filename)
        print(f"  Saved: {filename}")


if __name__ == "__main__":
    main()