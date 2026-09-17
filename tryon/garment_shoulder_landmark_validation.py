"""
Validates the garment shoulder landmark (topmost mask pixel outside
collar exclusion) against the garment's own contour structure.
Pure inspection — no deformation, no new transform.

Checks: does the shoulder point sit near a real contour feature
(a local curvature peak, consistent with a seam), or is it isolated
from any meaningful structure?

Run: python -m tryon.garment_shoulder_landmark_validation
"""

import io
import os
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw

from tryon.correspondence_shoulders_test import find_garment_shoulder_points, find_garment_underarms

GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
ALPHA_THRESHOLD = 10
CURVATURE_WINDOW = 8
OUTPUT_DIR = "outputs"


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD), image


def find_nearest_contour_point(contour, target):
    tx, ty = target
    best_idx, best_dist = None, float('inf')
    for i, pt in enumerate(contour):
        x, y = pt[0]
        d = np.hypot(x - tx, y - ty)
        if d < best_dist:
            best_dist, best_idx = d, i
    return best_idx, best_dist


def local_curvature_at(contour, idx, window=CURVATURE_WINDOW):
    n = len(contour)
    prev_i = (idx - window) % n
    next_i = (idx + window) % n
    p_prev = contour[prev_i][0].astype(float)
    p_curr = contour[idx][0].astype(float)
    p_next = contour[next_i][0].astype(float)
    v1 = p_curr - p_prev
    v2 = p_next - p_curr
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Fetching golden garment...")
    garment_resp = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{GOLDEN_GARMENT_ID}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    garment_bytes = requests.get(garment_resp["clean_image_url"]).content
    garment_mask, garment_img = get_binary_mask(garment_bytes)
    gw, gh = garment_img.size

    contours, _ = cv2.findContours(garment_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)

    left_shoulder, right_shoulder = find_garment_shoulder_points(garment_mask, gw)
    left_ua, right_ua = find_garment_underarms(garment_mask, gw)

    print(f"Shoulder points (proxy): L={left_shoulder}  R={right_shoulder}")
    print(f"Underarm points (validated earlier): L={left_ua}  R={right_ua}")

    print(f"\nValidating shoulder points against contour structure:")
    for name, pt in [("LEFT", left_shoulder), ("RIGHT", right_shoulder)]:
        idx, dist = find_nearest_contour_point(main_contour, pt)
        curvature = local_curvature_at(main_contour, idx)
        contour_pt = tuple(int(v) for v in main_contour[idx][0])
        print(f"  {name}: proxy={pt}  nearest_contour_pt={contour_pt}  "
              f"dist={dist:.1f}px  local_curvature={curvature:.1f}°")

    # For comparison, show curvature at the ALREADY-VALIDATED underarm 
    # points, as a reference for "what a real seam feature looks like"
    print(f"\nReference — curvature at already-validated underarm points:")
    for name, pt in [("LEFT_UA", left_ua), ("RIGHT_UA", right_ua)]:
        idx, dist = find_nearest_contour_point(main_contour, pt)
        curvature = local_curvature_at(main_contour, idx)
        print(f"  {name}: dist_to_contour={dist:.1f}px  local_curvature={curvature:.1f}°")

    # Visualize both sets of points directly on the garment with the 
    # full contour traced, for direct inspection
    canvas = garment_img.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    pts = [(int(p[0][0]), int(p[0][1])) for p in main_contour]
    draw.line(pts + [pts[0]], fill=(150, 150, 150), width=1)

    for pt, color, label in [
        (left_shoulder, (255, 0, 255), "L_shoulder(proxy)"),
        (right_shoulder, (255, 0, 255), "R_shoulder(proxy)"),
        (left_ua, (0, 255, 0), "L_underarm"),
        (right_ua, (0, 255, 0), "R_underarm"),
    ]:
        x, y = pt
        draw.ellipse([x-6, y-6, x+6, y+6], outline=color, width=3)
        draw.text((x+8, y-6), label, fill=color)

    canvas.save(f"{OUTPUT_DIR}/garment_shoulder_landmark_validation.png")
    print(f"\nSaved: {OUTPUT_DIR}/garment_shoulder_landmark_validation.png")
    print(f"\n>>> DECISION GATE: does the shoulder proxy's local curvature "
          f"resemble a real seam feature (comparable to the underarm "
          f"reference), or does it look like an arbitrary point with low "
          f"or inconsistent curvature — meaning we need a better shoulder "
          f"landmark before building the deformation mesh around it?")


if __name__ == "__main__":
    main()