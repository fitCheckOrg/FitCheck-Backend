"""
Computes local direction/curvature along the already-validated
right-side outer-contour walk (confirmed by direct visual inspection
to trace the real garment boundary accurately). Reports candidate
extrema for inspection — does NOT select or promote any point as
the shoulder landmark.

Question being tested: does the true shoulder junction correspond
to a distinctive geometric feature (a local curvature extremum) in
the contour itself, independent of surface orientation/texture?

Run: python -m tryon.right_contour_curvature_test
"""

import io
import os
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw

from tryon.correspondence_shoulders_test import find_garment_underarms
from tryon.garment_shoulder_region_extraction import get_binary_mask

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
PLAIN_SHIRT_ID = "7e16734e-6cb2-4254-adf5-4695549ad5db"
OUTPUT_DIR = "outputs"
STEP_PX = 6  # same resolution already confirmed sufficient
DIRECTION_WINDOW = 3  # points on each side for local direction estimate


def trace_outer_boundary(mask, start_pt, gw, step=STEP_PX):
    """Walks upward from start_pt along the outer (max-x, right side)
    boundary, returning a list of (y, x) points. Same tracing logic
    already visually validated in the candidate-landscape diagnostic."""
    x0, y0 = start_pt
    points = []
    for y in range(y0, 0, -step):
        row = mask[y]
        xs = np.where(row)[0]
        if len(xs) == 0:
            continue
        points.append((y, int(xs.max())))
    return points


def local_curvature(points, idx, window=DIRECTION_WINDOW):
    """Angle (in degrees) between the incoming and outgoing direction
    vectors at points[idx] — 0 = straight line, larger = sharper turn."""
    n = len(points)
    prev_i = max(0, idx - window)
    next_i = min(n - 1, idx + window)
    if prev_i == idx or next_i == idx:
        return None
    y1, x1 = points[prev_i]
    y2, x2 = points[idx]
    y3, x3 = points[next_i]
    v1 = np.array([x2 - x1, y2 - y1], dtype=float)
    v2 = np.array([x3 - x2, y3 - y2], dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    garment_resp = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{PLAIN_SHIRT_ID}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    garment_bytes = requests.get(garment_resp["clean_image_url"]).content
    garment_mask, garment_img = get_binary_mask(garment_bytes)
    gw, gh = garment_img.size

    left_ua, right_ua = find_garment_underarms(garment_mask, gw)
    print(f"Underarms: L={left_ua}  R={right_ua}")

    points = trace_outer_boundary(garment_mask, right_ua, gw)
    print(f"\nTraced {len(points)} points along right outer boundary\n")

    curvatures = []
    print(f"{'idx':<6}{'y':<6}{'x':<6}{'curvature_deg'}")
    print("-" * 40)
    for i, (y, x) in enumerate(points):
        c = local_curvature(points, i)
        curvatures.append(c)
        c_str = f"{c:.1f}" if c is not None else "N/A"
        print(f"{i:<6}{y:<6}{x:<6}{c_str}")

    # Find local maxima in curvature (candidate geometric transitions)
    valid = [(i, c) for i, c in enumerate(curvatures) if c is not None]
    print(f"\n--- Candidate curvature extrema (local maxima) ---")
    for i in range(1, len(valid) - 1):
        idx, c = valid[i]
        prev_c = valid[i-1][1]
        next_c = valid[i+1][1]
        if c > prev_c and c > next_c and c > 15:  # visible bend, not noise
            y, x = points[idx]
            print(f"  idx={idx}  y={y}  x={x}  curvature={c:.1f}°")

    # Visualize: full traced path, color-coded by curvature magnitude
    canvas = garment_img.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    max_c = max([c for c in curvatures if c is not None], default=1)
    for i, (y, x) in enumerate(points):
        c = curvatures[i]
        if c is None:
            color = (150, 150, 150)
        else:
            intensity = int(255 * min(c / max_c, 1.0))
            color = (intensity, 0, 255 - intensity)
        draw.ellipse([x-3, y-3, x+3, y+3], outline=color, width=2)
    x, y = right_ua
    draw.ellipse([x-6, y-6, x+6, y+6], outline=(0,255,0), width=3)
    draw.text((x+8, y-6), "R_UA", fill=(0,255,0))

    canvas.save(f"{OUTPUT_DIR}/right_contour_curvature_test.png")
    print(f"\nSaved: {OUTPUT_DIR}/right_contour_curvature_test.png")
    print(f"\n>>> Legend: blue=low curvature (straight), red=high curvature "
          f"(sharp bend). Look for: does one of the printed extrema line up "
          f"visually with the real shoulder/sleeve junction?")


if __name__ == "__main__":
    main()