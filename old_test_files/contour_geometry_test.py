"""
Contour-based garment shape extraction POC.

Tests one specific hypothesis: does the garment's OUTER SILHOUETTE
contain concave points (inward bends in the boundary) that align
with where sleeves visually separate from the torso — even though
the interior fabric is one connected mass (confirmed by the
connected-component test).

No new AI service. No Flutter. Uses only remove.bg's existing alpha
mask + OpenCV. Testing whether cv2.convexityDefects finds anything
useful — not assuming it will.

Run: python contour_geometry_test.py
"""

import sys
import io
import requests
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, ".")
from core.storage.supabase_client import supabase

TEST_CASES = [
    {"item_id": "25989827-db9d-4d65-90b3-12f5357b0b44", "label": "hanging-sleeve shirt"},
    {"item_id": "ac7fffb7-2e3a-40f8-894a-0e85fc245373", "label": "extended-sleeve polo"},
]

ALPHA_THRESHOLD = 10
MIN_DEFECT_DEPTH = 1000  # in OpenCV's fixed-point units (1/256 px) — 
                          # filters tiny noise-level notches from 
                          # real structural concavities


def get_binary_mask(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255


def find_contour_and_defects(binary_mask: np.ndarray):
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None, "No contour found at all"

    # Largest contour = the garment silhouette itself
    main_contour = max(contours, key=cv2.contourArea)

    hull_indices = cv2.convexHull(main_contour, returnPoints=False)

    try:
        defects = cv2.convexityDefects(main_contour, hull_indices)
    except cv2.error as e:
        return main_contour, None, f"convexityDefects failed: {e}"

    if defects is None:
        return main_contour, None, "No convexity defects found (silhouette is fully convex)"

    return main_contour, defects, None


def main():
    for case in TEST_CASES:
        print(f"\n{'='*60}")
        print(f"{case['item_id']} — {case['label']}")
        print('='*60)

        result = supabase.table("closet_items")\
            .select("clean_image_url").eq("item_id", case["item_id"]).single().execute()
        image_bytes = requests.get(result.data["clean_image_url"]).content

        mask = get_binary_mask(image_bytes)
        h, w = mask.shape
        print(f"  Image dimensions: {w}x{h}")

        contour, defects, error = find_contour_and_defects(mask)

        if error:
            print(f"  >>> {error}")
            continue

        print(f"  Contour has {len(contour)} points")
        print(f"  Total convexity defects found: {len(defects)}")

        # Filter to meaningful defects only, sorted by depth (most 
        # pronounced concavity first)
        significant = []
        defects_flat = defects.reshape(-1, 4)
        for i in range(len(defects_flat)):
            start_idx, end_idx, far_idx, depth = defects_flat[i]
            start_idx, end_idx, far_idx, depth = int(start_idx), int(end_idx), int(far_idx), int(depth)
            if depth > MIN_DEFECT_DEPTH:
                far_point = contour[far_idx][0]
                x_pct = round(far_point[0] / w * 100, 1)
                y_pct = round(far_point[1] / h * 100, 1)
                significant.append((depth, x_pct, y_pct))

        significant.sort(reverse=True)  # deepest first

        print(f"  Significant defects (depth > {MIN_DEFECT_DEPTH}): {len(significant)}")
        for depth, x_pct, y_pct in significant[:10]:  # top 10 max
            print(f"    depth={depth}  position=({x_pct}%, {y_pct}%)")

        if not significant:
            print("  >>> No significant concave points found — silhouette "
                  "is essentially convex/smooth at this depth threshold")


if __name__ == "__main__":
    main()