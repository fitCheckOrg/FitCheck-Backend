"""
Contour-based garment boundary visualization POC.

Takes the deepest convexity defect on EACH SIDE of the garment
(left half vs right half, separately — not just top-2-overall,
which could accidentally grab two points from the same side) as
candidate left/right underarm points. Draws them directly onto
the original garment image so we can visually confirm whether
they land at the real sleeve/torso boundary.

Does NOT modify GarmentRepresentationV2. Pure visual diagnostic —
proving whether contour-derived points are usable before building
any representation around them.

Run: python contour_visualization.py
"""

import sys
import io
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw

sys.path.insert(0, ".")
from core.storage.supabase_client import supabase

TEST_CASES = [
    {"item_id": "25989827-db9d-4d65-90b3-12f5357b0b44", "label": "hanging_sleeve_shirt"},
    {"item_id": "ac7fffb7-2e3a-40f8-894a-0e85fc245373", "label": "extended_sleeve_polo"},
]

ALPHA_THRESHOLD = 10
MIN_DEFECT_DEPTH = 1000


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255, image


def find_underarm_candidates(binary_mask):
    """Splits significant defects into left-half/right-half of the
    image, picks the deepest from EACH side separately — enforces
    getting one candidate per side, matching real garment bilateral
    structure, rather than risking two picks from the same side."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    main_contour = max(contours, key=cv2.contourArea)
    hull_indices = cv2.convexHull(main_contour, returnPoints=False)

    try:
        defects = cv2.convexityDefects(main_contour, hull_indices)
    except cv2.error:
        return main_contour, None, None

    if defects is None:
        return main_contour, None, None

    h_img, w_img = binary_mask.shape
    defects_flat = defects.reshape(-1, 4)

    left_candidates = []
    right_candidates = []

    for i in range(len(defects_flat)):
        start_idx, end_idx, far_idx, depth = defects_flat[i]
        depth = int(depth)
        far_idx = int(far_idx)
        if depth <= MIN_DEFECT_DEPTH:
            continue
        far_point = main_contour[far_idx][0]
        x_pct = far_point[0] / w_img * 100

        if x_pct < 50:
            left_candidates.append((depth, int(far_point[0]), int(far_point[1]), x_pct))
        else:
            right_candidates.append((depth, int(far_point[0]), int(far_point[1]), x_pct))

    left_candidates.sort(reverse=True)
    right_candidates.sort(reverse=True)

    left_best = left_candidates[0] if left_candidates else None
    right_best = right_candidates[0] if right_candidates else None

    return main_contour, left_best, right_best


def visualize(image, main_contour, left_point, right_point, w_img, h_img):
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    contour_points = [(int(p[0][0]), int(p[0][1])) for p in main_contour]
    draw.line(contour_points + [contour_points[0]], fill=(0, 100, 255), width=2)

    if left_point:
        _, lx, ly, _ = left_point
        draw.line([(lx, 0), (lx, h_img)], fill=(255, 200, 0), width=2)
        draw.ellipse([lx-8, ly-8, lx+8, ly+8], fill=(255, 0, 0))
        draw.text((lx+10, ly), "LEFT candidate", fill=(255, 0, 0))

    if right_point:
        _, rx, ry, _ = right_point
        draw.line([(rx, 0), (rx, h_img)], fill=(255, 200, 0), width=2)
        draw.ellipse([rx-8, ry-8, rx+8, ry+8], fill=(0, 255, 0))
        draw.text((rx+10, ry), "RIGHT candidate", fill=(0, 255, 0))

    return canvas


def main():
    for case in TEST_CASES:
        print(f"\n{'='*60}")
        print(f"{case['item_id']} — {case['label']}")
        print('='*60)

        result = supabase.table("closet_items")\
            .select("clean_image_url").eq("item_id", case["item_id"]).single().execute()
        image_bytes = requests.get(result.data["clean_image_url"]).content

        binary_mask, original_image = get_binary_mask(image_bytes)
        h_img, w_img = binary_mask.shape

        main_contour, left_point, right_point = find_underarm_candidates(binary_mask)

        if main_contour is None:
            print("  No contour found, skipping")
            continue

        if left_point:
            depth, lx, ly, x_pct = left_point
            print(f"  LEFT candidate:  depth={depth}  position=({x_pct:.1f}%, {ly/h_img*100:.1f}%)")
        else:
            print("  No left-side candidate found above threshold")

        if right_point:
            depth, rx, ry, x_pct = right_point
            print(f"  RIGHT candidate: depth={depth}  position=({x_pct:.1f}%, {ry/h_img*100:.1f}%)")
        else:
            print("  No right-side candidate found above threshold")

        annotated = visualize(original_image, main_contour, left_point, right_point, w_img, h_img)
        filename = f"contour_viz_{case['label']}.png"
        annotated.save(filename)
        print(f"  Saved: {filename}")


if __name__ == "__main__":
    main()