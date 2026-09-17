"""
Extracts the shoulder/sleeve structural boundary using the
orientation-discontinuity signal confirmed visually in
garment_region_structure_test.py.

Method:
1. Establish a REFERENCE torso orientation from patches well below
   the underarm line (unambiguously torso fabric).
2. Walk upward from the underarm along each side (near the outer
   garment edge), classifying each patch as "matches torso
   reference" or "deviates" (sleeve-like).
3. Report the y-coordinate where a SUSTAINED deviation begins
   (not a single flicker) — the shoulder/sleeve junction candidate.

Run: python -m tryon.garment_shoulder_region_extraction
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
GRID_SIZE = 12
REFERENCE_Y_BELOW_UA = 60   # how far below underarm to sample reference torso patches
DEVIATION_THRESHOLD_DEG = 20  # angle difference to count as "deviates"
SUSTAINED_COUNT = 3  # consecutive deviating patches required before declaring transition
OUTPUT_DIR = "outputs"


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD), image


def local_dominant_orientation(gray_patch):
    if gray_patch.size == 0 or gray_patch.shape[0] < 3 or gray_patch.shape[1] < 3:
        return None, 0.0
    gx = cv2.Sobel(gray_patch, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_patch, cv2.CV_64F, 0, 1, ksize=3)
    Jxx, Jyy, Jxy = np.sum(gx*gx), np.sum(gy*gy), np.sum(gx*gy)
    angle = 0.5 * np.arctan2(2*Jxy, Jxx - Jyy)
    angle_deg = np.degrees(angle) % 180
    trace = Jxx + Jyy
    if trace == 0:
        return None, 0.0
    diff = np.sqrt((Jxx-Jyy)**2 + 4*Jxy**2)
    coherence = diff / trace
    return angle_deg, coherence


def angle_diff(a, b):
    """Smallest difference between two orientations on a 0-180 circle."""
    d = abs(a - b) % 180
    return min(d, 180 - d)


def get_reference_orientation(gray, mask, underarm_y, gw):
    """Samples patches in a band well below the underarm, across the
    full torso width, and returns the median orientation."""
    y_start = underarm_y + REFERENCE_Y_BELOW_UA
    y_end = y_start + 40
    angles = []
    for y in range(y_start, y_end, GRID_SIZE):
        for x in range(0, gw, GRID_SIZE):
            patch_mask = mask[y:y+GRID_SIZE, x:x+GRID_SIZE]
            if patch_mask.sum() < (GRID_SIZE*GRID_SIZE*0.5):
                continue
            gray_patch = gray[y:y+GRID_SIZE, x:x+GRID_SIZE]
            angle, coherence = local_dominant_orientation(gray_patch)
            if angle is not None and coherence >= 0.3:
                angles.append(angle)
    return np.median(angles) if angles else None


def walk_side(gray, mask, underarm_pt, reference_angle, gw, side):
    """Walks upward from the underarm, staying near the outer garment
    edge on this side, classifying each row-band as matching or 
    deviating from the reference torso orientation."""
    ua_x, ua_y = underarm_pt
    results = []

    for y in range(ua_y, 0, -GRID_SIZE):
        row_mask = mask[y]
        xs = np.where(row_mask)[0]
        if len(xs) == 0:
            continue
        outer_x = xs.min() if side == "left" else xs.max()
        x_start = max(0, outer_x - GRID_SIZE) if side == "left" else outer_x
        x_end = outer_x if side == "left" else min(gw, outer_x + GRID_SIZE)

        patch_mask = mask[y:y+GRID_SIZE, x_start:x_end]
        gray_patch = gray[y:y+GRID_SIZE, x_start:x_end]
        if patch_mask.sum() < (patch_mask.size * 0.3):
            results.append((y, outer_x, None, None))
            continue

        angle, coherence = local_dominant_orientation(gray_patch)
        if angle is None or coherence < 0.25:
            results.append((y, outer_x, None, None))
            continue
        
        deviates = bool(angle_diff(angle, reference_angle) > DEVIATION_THRESHOLD_DEG)
        results.append((y, outer_x, angle, deviates))

    return results


def find_sustained_transition(walk_results, max_gap=2):
    """
    Finds the first point where SUSTAINED_COUNT consecutive True
    (deviating) results occur, now tolerating up to max_gap
    intervening False (matching) results without fully resetting —
    a single noisy row shouldn't erase real, surrounding signal.
    None (unclassified/low-coherence) rows still don't count as a
    gap or a match; they're skipped through either way.
    """
    consecutive = 0
    gap_count = 0
    start_idx = None
    for i, (y, x, angle, deviates) in enumerate(walk_results):
        if deviates is True:
            if consecutive == 0:
                start_idx = i
            consecutive += 1
            gap_count = 0
            if consecutive >= SUSTAINED_COUNT:
                return walk_results[start_idx]
        elif deviates is False:
            gap_count += 1
            if gap_count > max_gap:
                consecutive = 0
                gap_count = 0
                start_idx = None
        # None: unchanged, skip through
    return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Fetching golden garment...")
    garment_resp = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{GOLDEN_GARMENT_ID}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    garment_bytes = requests.get(garment_resp["clean_image_url"]).content
    garment_mask, garment_img = get_binary_mask(garment_bytes)
    gw, gh = garment_img.size
    gray = cv2.cvtColor(np.array(garment_img.convert("RGB")), cv2.COLOR_RGB2GRAY)

    left_ua, right_ua = find_garment_underarms(garment_mask, gw)
    print(f"Underarms: L={left_ua}  R={right_ua}")

    underarm_y = max(left_ua[1], right_ua[1])
    reference_angle = get_reference_orientation(gray, garment_mask, underarm_y, gw)
    print(f"\nReference torso orientation: {reference_angle:.1f}°")

    left_walk = walk_side(gray, garment_mask, left_ua, reference_angle, gw, "left")
    right_walk = walk_side(gray, garment_mask, right_ua, reference_angle, gw, "right")

    print(f"\n--- LEFT side walk (underarm upward) ---")
    for y, x, angle, deviates in left_walk:
        angle_str = f"{angle:.1f}°" if angle is not None else "N/A"
        dev_str = "DEVIATES" if deviates else ("matches" if deviates is False else "skip")
        print(f"  y={y:4d}  x={x:4d}  angle={angle_str:>7}  {dev_str}")

    print(f"\n--- RIGHT side walk (underarm upward) ---")
    for y, x, angle, deviates in right_walk:
        angle_str = f"{angle:.1f}°" if angle is not None else "N/A"
        dev_str = "DEVIATES" if deviates else ("matches" if deviates is False else "skip")
        print(f"  y={y:4d}  x={x:4d}  angle={angle_str:>7}  {dev_str}")

    left_transition = find_sustained_transition(left_walk)
    right_transition = find_sustained_transition(right_walk)

    print(f"\nLEFT sustained transition: {left_transition}")
    print(f"RIGHT sustained transition: {right_transition}")

    canvas = garment_img.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    for pt, color, label in [(left_ua, (0,255,0), "L_UA"), (right_ua, (0,255,0), "R_UA")]:
        x, y = pt
        draw.ellipse([x-6,y-6,x+6,y+6], outline=color, width=3)
        draw.text((x+8,y-6), label, fill=color)

    for transition, color, label in [
        (left_transition, (255,255,0), "L_shoulder(region-based)"),
        (right_transition, (255,0,255), "R_shoulder(region-based)"),
    ]:
        if transition is None:
            continue
        y, x, angle, deviates = transition
        draw.ellipse([x-8,y-8,x+8,y+8], outline=color, width=3)
        draw.text((x+10,y-8), label, fill=color)

    canvas.save(f"{OUTPUT_DIR}/garment_shoulder_region_extraction.png")
    print(f"\nSaved: {OUTPUT_DIR}/garment_shoulder_region_extraction.png")
    print(f"\n>>> DECISION GATE: does the region-based candidate land at the "
          f"real shoulder/sleeve seam, closer to the true structural "
          f"junction than either previous contour-based attempt?")


if __name__ == "__main__":
    main()