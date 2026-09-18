"""
Garment silhouette/landmark broad validation.

Tests the candidate V3 primitives (silhouette, underarms, collar,
hem) against a wider, more structurally varied real sample — not
another algorithm experiment, a reliability check on the mechanism
already proven to work on 2 garments.

Does NOT modify the schema. Pure validation.

Run: python broad_landmark_validation.py
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


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255, image


def extract_landmarks(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, None, None

    main_contour = max(contours, key=cv2.contourArea)
    h_img, w_img = binary_mask.shape

    # Underarms — deepest defect on each half
    hull_indices = cv2.convexHull(main_contour, returnPoints=False)
    left_point = right_point = None
    try:
        defects = cv2.convexityDefects(main_contour, hull_indices)
        if defects is not None:
            defects_flat = defects.reshape(-1, 4)
            left_candidates, right_candidates = [], []
            for i in range(len(defects_flat)):
                _, _, far_idx, depth = defects_flat[i]
                depth, far_idx = int(depth), int(far_idx)
                if depth <= MIN_DEFECT_DEPTH:
                    continue
                far_pt = main_contour[far_idx][0]
                x_pct = far_pt[0] / w_img * 100
                entry = (depth, int(far_pt[0]), int(far_pt[1]))
                (left_candidates if x_pct < 50 else right_candidates).append(entry)
            left_candidates.sort(reverse=True)
            right_candidates.sort(reverse=True)
            left_point = left_candidates[0][1:] if left_candidates else None
            right_point = right_candidates[0][1:] if right_candidates else None
    except cv2.error:
        pass

    # Hem — lowest contour point
    lowest_idx = np.argmax(main_contour[:, 0, 1])
    hem_point = tuple(int(v) for v in main_contour[lowest_idx][0])

    # Collar — topmost contour point (candidate proxy, not GPT here — 
    # pure-geometry version for consistency with everything else 
    # in this test)
    topmost_idx = np.argmin(main_contour[:, 0, 1])
    collar_point = tuple(int(v) for v in main_contour[topmost_idx][0])

    return main_contour, left_point, right_point, hem_point, collar_point


def visualize(image, contour, left, right, hem, collar):
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    pts = [(int(p[0][0]), int(p[0][1])) for p in contour]
    draw.line(pts + [pts[0]], fill=(0, 100, 255), width=2)

    for point, color, label in [
        (left, (255, 0, 0), "L-underarm"),
        (right, (0, 255, 0), "R-underarm"),
        (hem, (255, 0, 255), "hem"),
        (collar, (0, 255, 255), "collar"),
    ]:
        if point:
            x, y = point
            draw.ellipse([x-8, y-8, x+8, y+8], fill=color)
            draw.text((x+10, y), label, fill=color)

    return canvas


def main():
    result = supabase.table("closet_items")\
        .select("item_id, item_type, category, clean_image_url")\
        .eq("category", "top")\
        .eq("is_archived", False)\
        .limit(9)\
        .execute()

    items = result.data or []
    print(f"Testing {len(items)} real items...\n")

    for item in items:
        print(f"\n{'='*60}")
        print(f"{item['item_id']} — {item['item_type']}")
        print('='*60)

        image_bytes = requests.get(item["clean_image_url"]).content
        mask, original = get_binary_mask(image_bytes)
        h_img, w_img = mask.shape

        contour, left, right, hem, collar = extract_landmarks(mask)
        if contour is None:
            print("  No contour found")
            continue

        for name, pt in [("left_underarm", left), ("right_underarm", right),
                          ("hem", hem), ("collar", collar)]:
            if pt:
                x_pct = round(pt[0] / w_img * 100, 1)
                y_pct = round(pt[1] / h_img * 100, 1)
                print(f"  {name}: ({x_pct}%, {y_pct}%)")
            else:
                print(f"  {name}: NOT FOUND")

        annotated = visualize(original, contour, left, right, hem, collar)
        filename = f"landmark_test_{item['item_id'][:8]}.png"
        annotated.save(filename)
        print(f"  Saved: {filename}")


if __name__ == "__main__":
    main()