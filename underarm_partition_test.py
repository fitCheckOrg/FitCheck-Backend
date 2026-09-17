"""
Underarm-partitioned contour arc POC.

Tests whether the validated left/right underarm points can PARTITION
the closed contour into two meaningful arcs — upper (collar
candidate) and lower (hem candidate) — rather than relying on
global topmost/lowest pixel extremes (both now closed: collar
failed on hanger hardware, hem failed on sleeves extending past
torso length).

Does NOT declare either candidate correct. Pure visualization —
does the partition itself look geometrically sensible.

Run: python underarm_partition_test.py
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

# Real cases already in hand — including both known failures
TEST_CASES = [
    {"item_id": "25989827-db9d-4d65-90b3-12f5357b0b44", "label": "hanger_shirt_collar_failure"},
    {"item_id": "ba2e5aa1-bfb5-462f-aaad-fe378f658c42", "label": "long_sleeve_hem_failure"},
    {"item_id": "ac7fffb7-2e3a-40f8-894a-0e85fc245373", "label": "polo_baseline"},
]


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255, image


def find_underarm_indices(main_contour, w_img):
    """Same validated mechanism as before, but this time we need
    the actual CONTOUR INDEX of each underarm point, not just its
    pixel position — required to slice the contour into arcs."""
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


def split_contour_arcs(main_contour, left_idx, right_idx):
    """
    Splits the closed contour into two arcs at the underarm indices.
    Contour is ordered (either clockwise or counter-clockwise) —
    slicing between two indices gives one arc, the remainder gives
    the other. We identify which is "upper" vs "lower" by checking
    which arc has the lower average y-coordinate (upper = smaller y).
    """
    n = len(main_contour)
    lo, hi = sorted([left_idx, right_idx])

    arc_a = main_contour[lo:hi+1]
    arc_b = np.concatenate([main_contour[hi:], main_contour[:lo+1]])

    avg_y_a = np.mean(arc_a[:, 0, 1])
    avg_y_b = np.mean(arc_b[:, 0, 1])

    if avg_y_a < avg_y_b:
        return arc_a, arc_b  # a=upper, b=lower
    else:
        return arc_b, arc_a


def visualize(image, main_contour, left_idx, right_idx, upper_arc, lower_arc):
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    # Full contour, faint
    full_pts = [(int(p[0][0]), int(p[0][1])) for p in main_contour]
    draw.line(full_pts + [full_pts[0]], fill=(80, 80, 80), width=1)

    # Upper arc — collar candidate region
    upper_pts = [(int(p[0][0]), int(p[0][1])) for p in upper_arc]
    if len(upper_pts) > 1:
        draw.line(upper_pts, fill=(0, 255, 255), width=3)

    # Lower arc — hem candidate region
    lower_pts = [(int(p[0][0]), int(p[0][1])) for p in lower_arc]
    if len(lower_pts) > 1:
        draw.line(lower_pts, fill=(255, 0, 255), width=3)

    # Underarm points themselves
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

        upper_arc, lower_arc = split_contour_arcs(main_contour, left_idx, right_idx)

        print(f"  Upper arc: {len(upper_arc)} points (cyan)")
        print(f"  Lower arc: {len(lower_arc)} points (magenta)")

        # Add this right after computing upper_arc, lower_arc in main(), 
# before the visualize() call:

        upper_y_min = np.min(upper_arc[:, 0, 1]) / h_img * 100
        upper_y_max = np.max(upper_arc[:, 0, 1]) / h_img * 100
        lower_y_min = np.min(lower_arc[:, 0, 1]) / h_img * 100
        lower_y_max = np.max(lower_arc[:, 0, 1]) / h_img * 100

        print(f"  Upper arc y-range: {upper_y_min:.1f}% to {upper_y_max:.1f}%")
        print(f"  Lower arc y-range: {lower_y_min:.1f}% to {lower_y_max:.1f}%")
        annotated = visualize(original, main_contour, left_idx, right_idx, upper_arc, lower_arc)
        filename = f"partition_{case['label']}.png"
        annotated.save(filename)
        print(f"  Saved: {filename}")


if __name__ == "__main__":
    main()


