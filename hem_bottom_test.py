"""
Narrow validation test: is the lowest point of the garment's outer
contour reliably the torso hem, or could it accidentally be a
sleeve tip, dangling fabric, or other artifact?

Reuses the exact same contour-extraction code already validated —
no new technique, just checking one specific point's real meaning.

Run: python hem_bottom_test.py
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


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255, image


def find_lowest_contour_point(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    main_contour = max(contours, key=cv2.contourArea)

    # Find the point with the maximum y-coordinate (lowest on screen,
    # since image y increases downward)
    lowest_idx = np.argmax(main_contour[:, 0, 1])
    lowest_point = main_contour[lowest_idx][0]

    return main_contour, (int(lowest_point[0]), int(lowest_point[1]))


def visualize(image, main_contour, lowest_point):
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    contour_points = [(int(p[0][0]), int(p[0][1])) for p in main_contour]
    draw.line(contour_points + [contour_points[0]], fill=(0, 100, 255), width=2)

    if lowest_point:
        lx, ly = lowest_point
        draw.ellipse([lx-10, ly-10, lx+10, ly+10], fill=(255, 0, 255))
        draw.line([(0, ly), (canvas.width, ly)], fill=(255, 255, 0), width=1)
        draw.text((lx+15, ly-10), "LOWEST POINT", fill=(255, 0, 255))

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

        main_contour, lowest_point = find_lowest_contour_point(binary_mask)

        if lowest_point is None:
            print("  No contour found")
            continue

        lx, ly = lowest_point
        x_pct = round(lx / w_img * 100, 1)
        y_pct = round(ly / h_img * 100, 1)
        print(f"  Lowest contour point: ({x_pct}%, {y_pct}%)")

        annotated = visualize(original_image, main_contour, lowest_point)
        filename = f"hem_test_{case['label']}.png"
        annotated.save(filename)
        print(f"  Saved: {filename}")


if __name__ == "__main__":
    main()