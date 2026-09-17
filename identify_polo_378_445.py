"""
Identifies what (378,445) actually is on the polo garment — the
point that won as the top curvature candidate on BOTH LEFT and
RIGHT searches, independent of underarm-collision (already ruled
out). Checks its position against the polo's own image dimensions
and against the independently-confirmed hem-corner percentages
from much earlier tonight's investigation.

Run: python identify_polo_378_445.py
"""

import io
import requests
import numpy as np
import cv2
from PIL import Image

ALPHA_THRESHOLD = 10
ITEM_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
POINT = (378, 445)


def main():
    result_data = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{ITEM_ID}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    image_bytes = requests.get(result_data["clean_image_url"]).content

    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    binary_mask = (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255
    h_img, w_img = binary_mask.shape

    x, y = POINT
    x_pct = x / w_img * 100
    y_pct = y / h_img * 100

    print(f"Image dimensions: {w_img} x {h_img}")
    print(f"Point {POINT} → ({x_pct:.1f}%, {y_pct:.1f}%) of image")

    # Independently-confirmed hem corner percentages from earlier 
    # tonight's separate investigation (visually verified against 
    # the actual photo at the time)
    known_hem_corners_pct = [
        {"label": "A1 (right hem corner)", "x_pct": 80.3, "y_pct": 84.0},
        {"label": "A2 (left hem corner)", "x_pct": 19.7, "y_pct": 83.4},
    ]

    print(f"\nComparing against independently-confirmed hem corners "
          f"(from earlier visual-verification session):")
    for hem in known_hem_corners_pct:
        dx = abs(x_pct - hem["x_pct"])
        dy = abs(y_pct - hem["y_pct"])
        print(f"  {hem['label']}: ({hem['x_pct']}%, {hem['y_pct']}%)  "
              f"— diff: ({dx:.1f}pp, {dy:.1f}pp)")

    # Also find the actual lowest contour point for direct comparison
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)
    lowest_idx = np.argmax(main_contour[:, 0, 1])
    lowest_pt = tuple(int(v) for v in main_contour[lowest_idx][0])
    lowest_pct = (lowest_pt[0] / w_img * 100, lowest_pt[1] / h_img * 100)
    dist_to_lowest = np.hypot(x - lowest_pt[0], y - lowest_pt[1])
    print(f"\nActual lowest contour point (hem, by definition): {lowest_pt} "
          f"({lowest_pct[0]:.1f}%, {lowest_pct[1]:.1f}%)")
    print(f"Distance from {POINT} to actual lowest point: {dist_to_lowest:.1f}px")


if __name__ == "__main__":
    main()