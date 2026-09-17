"""
Diagnostic: does the garment image contain a usable structural
signal (independent of stripe COLOR) that could distinguish sleeve
fabric from torso fabric? Tests local stripe/edge orientation via
gradient structure tensor — a real physical signal (seam-adjacent
fabric panels are often cut/attached at slightly different angles),
not a color heuristic.

Pure diagnostic. Visualizes candidate signal only. No landmark
extraction, no correspondence, no deformation.

Run: python -m tryon.garment_region_structure_test
"""

import io
import os
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw

from tryon.correspondence_shoulders_test import find_garment_underarms

GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
ALPHA_THRESHOLD = 10
GRID_SIZE = 12  # patch size in pixels for local orientation sampling
OUTPUT_DIR = "outputs"


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD), image


def local_dominant_orientation(gray_patch):
    """
    Computes the dominant local edge/stripe orientation in a patch
    via the structure tensor: builds Ix, Iy gradients, then finds
    the eigenvector direction of [[Ix², IxIy],[IxIy, Iy²]] summed
    over the patch. Returns angle in degrees (0=horizontal,
    90=vertical) and a coherence score (0-1, how strongly directional
    the patch is — low coherence = noisy/no clear stripe direction).
    """
    if gray_patch.size == 0 or gray_patch.shape[0] < 3 or gray_patch.shape[1] < 3:
        return None, 0.0

    gx = cv2.Sobel(gray_patch, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_patch, cv2.CV_64F, 0, 1, ksize=3)

    Jxx = np.sum(gx * gx)
    Jyy = np.sum(gy * gy)
    Jxy = np.sum(gx * gy)

    # Dominant orientation is perpendicular to gradient direction 
    # (gradient points across stripes; stripe direction is along them)
    angle = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)
    angle_deg = np.degrees(angle) % 180

    # Coherence: how anisotropic the gradient distribution is
    trace = Jxx + Jyy
    if trace == 0:
        return None, 0.0
    diff = np.sqrt((Jxx - Jyy)**2 + 4 * Jxy**2)
    coherence = diff / trace

    return angle_deg, coherence


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Fetching golden garment...")
    garment_resp = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{GOLDEN_GARMENT_ID}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    garment_bytes = requests.get(garment_resp["clean_image_url"]).content
    garment_mask, garment_img = get_binary_mask(garment_bytes)
    gw, gh = garment_img.size

    left_ua, right_ua = find_garment_underarms(garment_mask, gw)
    print(f"Underarms (trusted reference): L={left_ua}  R={right_ua}")

    gray = cv2.cvtColor(np.array(garment_img.convert("RGB")), cv2.COLOR_RGB2GRAY)

    # Region of interest: from top of garment down to underarm level, 
    # full width — this is where the sleeve/shoulder question lives
    underarm_y = max(left_ua[1], right_ua[1])
    roi_top = 0
    roi_bottom = underarm_y + 20  # small margin past underarm for context

    print(f"\nAnalyzing region y=[{roi_top}, {roi_bottom}] in {GRID_SIZE}px patches...")

    canvas = garment_img.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    orientations_found = 0
    for y in range(roi_top, roi_bottom, GRID_SIZE):
        for x in range(0, gw, GRID_SIZE):
            patch_mask = garment_mask[y:y+GRID_SIZE, x:x+GRID_SIZE]
            if patch_mask.sum() < (GRID_SIZE * GRID_SIZE * 0.5):
                continue  # skip patches mostly outside the garment

            gray_patch = gray[y:y+GRID_SIZE, x:x+GRID_SIZE]
            angle, coherence = local_dominant_orientation(gray_patch)

            if angle is None or coherence < 0.3:
                continue  # skip low-confidence patches, don't force a guess

            orientations_found += 1
            cx, cy = x + GRID_SIZE // 2, y + GRID_SIZE // 2
            length = GRID_SIZE * 0.6 * coherence
            angle_rad = np.radians(angle)
            dx = length * np.cos(angle_rad)
            dy = length * np.sin(angle_rad)

            # Color-code by angle for visual grouping: similar angles 
            # get similar colors, making discontinuities visible
            hue = int((angle / 180.0) * 179)
            color_hsv = np.uint8([[[hue, 255, 255]]])
            color_rgb = tuple(int(c) for c in cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0][0])

            draw.line([cx-dx, cy-dy, cx+dx, cy+dy], fill=color_rgb, width=2)

    print(f"Patches with confident orientation: {orientations_found}")

    for pt, label in [(left_ua, "L_UA"), (right_ua, "R_UA")]:
        x, y = pt
        draw.ellipse([x-6, y-6, x+6, y+6], outline=(0, 255, 0), width=3)
        draw.text((x+8, y-6), label, fill=(0, 255, 0))

    canvas.save(f"{OUTPUT_DIR}/garment_region_structure_test.png")
    print(f"\nSaved: {OUTPUT_DIR}/garment_region_structure_test.png")
    print(f"\n>>> DECISION GATE: do the colored orientation lines show a "
          f"visible, coherent discontinuity between the sleeve regions "
          f"and the torso/body region (a real angle change near the "
          f"shoulder), or is the orientation pattern uniform/noisy "
          f"throughout — meaning this signal isn't usable either?")


if __name__ == "__main__":
    main()