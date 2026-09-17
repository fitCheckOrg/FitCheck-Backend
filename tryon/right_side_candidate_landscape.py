"""
Maps the full right-side orientation landscape at GRID_SIZE=6,
alongside a directly-marked visual estimate of the TRUE shoulder
location (to be placed by eye from the saved image, then checked
against what signal actually exists there).

Does not change the detector. Does not touch the mesh. Pure
candidate-landscape diagnostic, per explicit scope.

Run: python -m tryon.right_side_candidate_landscape
"""

import io
import os
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw

from tryon.correspondence_shoulders_test import find_garment_underarms
from tryon.garment_shoulder_region_extraction import local_dominant_orientation, angle_diff, get_binary_mask

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
PLAIN_SHIRT_ID = "7e16734e-6cb2-4254-adf5-4695549ad5db"
OUTPUT_DIR = "outputs"
GRID_SIZE = 6
DEVIATION_THRESHOLD_DEG = 20
REFERENCE_Y_BELOW_UA = 60

# Rough visual estimate of the true right shoulder seam location —
# a first guess to check signal at, NOT assumed correct. Placed
# roughly where the right collar/shoulder-top region visually sits,
# based on the underarm and the general garment proportions already
# seen on other garments tonight. Adjust after viewing the printed
# landscape if this guess looks off.
ESTIMATED_TRUE_SHOULDER_RIGHT = (470, 60)


def get_reference_orientation_g(gray, mask, underarm_y, gw):
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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    garment_resp = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{PLAIN_SHIRT_ID}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    garment_bytes = requests.get(garment_resp["clean_image_url"]).content
    garment_mask, garment_img = get_binary_mask(garment_bytes)
    gw, gh = garment_img.size
    gray = cv2.cvtColor(np.array(garment_img.convert("RGB")), cv2.COLOR_RGB2GRAY)

    left_ua, right_ua = find_garment_underarms(garment_mask, gw)
    print(f"Underarms: L={left_ua}  R={right_ua}")

    underarm_y = max(left_ua[1], right_ua[1])
    reference_angle = get_reference_orientation_g(gray, garment_mask, underarm_y, gw)
    print(f"Reference torso orientation: {reference_angle:.1f}°\n")

    # Full landscape: EVERY patch in the right-side region, not just 
    # the narrow outer-edge walk — scan a full band from underarm to 
    # collar, across a range of x near the outer edge AND inward
    print(f"{'y':<6}{'x':<6}{'angle':<10}{'coherence':<12}{'deviates'}")
    print("-" * 50)

    ua_x, ua_y = right_ua
    landscape = []
    for y in range(ua_y, 0, -GRID_SIZE):
        row_mask = garment_mask[y]
        xs = np.where(row_mask)[0]
        if len(xs) == 0:
            continue
        outer_x = xs.max()
        # Scan several patches inward from the outer edge, not just one
        for depth in range(0, 4):
            x_end = min(gw, outer_x - depth * GRID_SIZE)
            x_start = max(0, x_end - GRID_SIZE)
            if x_start >= x_end:
                continue
            patch_mask = garment_mask[y:y+GRID_SIZE, x_start:x_end]
            if patch_mask.sum() < (patch_mask.size * 0.3):
                continue
            gray_patch = gray[y:y+GRID_SIZE, x_start:x_end]
            angle, coherence = local_dominant_orientation(gray_patch)
            if angle is None:
                continue
            deviates = angle_diff(angle, reference_angle) > DEVIATION_THRESHOLD_DEG if coherence >= 0.25 else None
            landscape.append((y, x_start, angle, coherence, deviates))
            print(f"{y:<6}{x_start:<6}{angle:<10.1f}{coherence:<12.3f}{deviates}")

    # Check signal specifically at the estimated true shoulder location
    est_x, est_y = ESTIMATED_TRUE_SHOULDER_RIGHT
    print(f"\n--- Signal at estimated true shoulder ({est_x},{est_y}) ---")
    patch = gray[est_y:est_y+GRID_SIZE, est_x:est_x+GRID_SIZE]
    angle, coherence = local_dominant_orientation(patch)
    if angle is not None:
        deviates = angle_diff(angle, reference_angle) > DEVIATION_THRESHOLD_DEG
        print(f"angle={angle:.1f}°  coherence={coherence:.3f}  deviates={deviates}")
    else:
        print("No usable patch here (insufficient garment coverage or size)")

    # Visualize the full landscape + the estimated true point
    canvas = garment_img.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for y, x, angle, coherence, deviates in landscape:
        color = (255,80,80) if deviates is True else (0,200,0) if deviates is False else (150,150,150)
        draw.ellipse([x-3,y-3,x+3,y+3], outline=color, width=2)
    draw.ellipse([est_x-8,est_y-8,est_x+8,est_y+8], outline=(0,255,255), width=3)
    draw.text((est_x+10,est_y-8), "estimated_true_shoulder", fill=(0,255,255))
    x,y = right_ua
    draw.ellipse([x-6,y-6,x+6,y+6], outline=(0,150,255), width=3)
    draw.text((x+8,y-6), "R_UA", fill=(0,150,255))

    canvas.save(f"{OUTPUT_DIR}/right_side_candidate_landscape.png")
    print(f"\nSaved: {OUTPUT_DIR}/right_side_candidate_landscape.png")
    print(f"\n>>> Check: does the estimated true shoulder location show real, "
          f"strong deviating signal (comparable to the false-positive sleeve-"
          f"edge point), or weak/no signal — telling us whether this is a "
          f"'multiple real signals, wrong one selected' problem or a "
          f"'true location has no signal at all' problem?")


if __name__ == "__main__":
    main()