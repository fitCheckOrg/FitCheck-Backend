"""
Raw contour geometry inspection around A#1's strong event and its
expected opposite-side region.

Not another classifier, not another threshold. Direct visual +
numeric inspection of the actual pixel-level contour shape at two
places: where the strong 98.7° event genuinely is, and where its
theoretically-expected counterpart would sit if the garment were
symmetric — to see what's ACTUALLY there instead of assuming any
of the six candidate explanations.

Run: python raw_geometry_inspection.py
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

ITEM_ID = "bb7763f6-f1df-4b26-9522-94299149bb5e"

# A#1's confirmed real position
KNOWN_EVENT_X_PCT = 10.4
KNOWN_EVENT_Y_PCT = 96.4

# Mirror position — where a symmetric counterpart would be expected
EXPECTED_MIRROR_X_PCT = 100 - KNOWN_EVENT_X_PCT
EXPECTED_MIRROR_Y_PCT = KNOWN_EVENT_Y_PCT


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255, image


def get_main_contour(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea)


def get_local_contour_window(main_contour, target_x_pct, target_y_pct, w_img, h_img, window_size=25):
    """Finds the contour point closest to the target position, then
    returns a window of raw points around it — the actual local
    shape, not a summary statistic."""
    target_x = target_x_pct / 100 * w_img
    target_y = target_y_pct / 100 * h_img

    pts = main_contour[:, 0, :].astype(float)
    distances = np.hypot(pts[:, 0] - target_x, pts[:, 1] - target_y)
    closest_idx = np.argmin(distances)
    closest_dist = distances[closest_idx]

    n = len(main_contour)
    window_indices = [(closest_idx + i) % n for i in range(-window_size, window_size + 1)]
    window_points = [tuple(int(v) for v in main_contour[idx][0]) for idx in window_indices]

    return closest_idx, closest_dist, window_points


def visualize_regions(image, region_a_points, region_b_points, w_img, h_img):
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    # Region A (known strong event) — bright cyan, thick
    if len(region_a_points) > 1:
        draw.line(region_a_points, fill=(0, 255, 255), width=4)
    # Region B (expected mirror) — bright magenta, thick
    if len(region_b_points) > 1:
        draw.line(region_b_points, fill=(255, 0, 255), width=4)

    return canvas


def main():
    result = supabase.table("closet_items")\
        .select("clean_image_url, item_type").eq("item_id", ITEM_ID).single().execute()
    image_bytes = requests.get(result.data["clean_image_url"]).content

    print(f"item_type: {result.data['item_type']}")

    binary_mask, original = get_binary_mask(image_bytes)
    h_img, w_img = binary_mask.shape
    main_contour = get_main_contour(binary_mask)

    print(f"\n--- Region A: known strong event ({KNOWN_EVENT_X_PCT}%, {KNOWN_EVENT_Y_PCT}%) ---")
    idx_a, dist_a, window_a = get_local_contour_window(
        main_contour, KNOWN_EVENT_X_PCT, KNOWN_EVENT_Y_PCT, w_img, h_img
    )
    print(f"Closest contour index: {idx_a}, distance from target: {dist_a:.1f}px")
    print("Local raw points (x_pct, y_pct):")
    for pt in window_a[::5]:  # every 5th point, full window would be too dense to read
        print(f"  ({pt[0]/w_img*100:.1f}%, {pt[1]/h_img*100:.1f}%)")

    print(f"\n--- Region B: expected mirror position ({EXPECTED_MIRROR_X_PCT}%, {EXPECTED_MIRROR_Y_PCT}%) ---")
    idx_b, dist_b, window_b = get_local_contour_window(
        main_contour, EXPECTED_MIRROR_X_PCT, EXPECTED_MIRROR_Y_PCT, w_img, h_img
    )
    print(f"Closest contour index: {idx_b}, distance from target: {dist_b:.1f}px")
    print("Local raw points (x_pct, y_pct):")
    for pt in window_b[::5]:
        print(f"  ({pt[0]/w_img*100:.1f}%, {pt[1]/h_img*100:.1f}%)")

    # Bounding box comparison — does one side extend further than the other?
    xs = main_contour[:, 0, 0]
    ys = main_contour[:, 0, 1]
    print(f"\n--- Overall silhouette bounds ---")
    print(f"x range: {xs.min()/w_img*100:.1f}% to {xs.max()/w_img*100:.1f}%")
    print(f"y range: {ys.min()/h_img*100:.1f}% to {ys.max()/h_img*100:.1f}%")

    canvas = visualize_regions(original, window_a, window_b, w_img, h_img)
    canvas.save("raw_geometry_inspection.png")
    print(f"\nSaved: raw_geometry_inspection.png (cyan=Region A, magenta=Region B)")


if __name__ == "__main__":
    main()