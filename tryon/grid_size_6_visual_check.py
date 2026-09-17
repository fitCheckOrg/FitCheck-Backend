"""
Pure visual confirmation of GRID_SIZE=6 detection on the plain
shirt. Draws every sampled walk point (color-coded True/False/None)
with the actual returned transition point highlighted separately.
Does NOT modify the detector. Does NOT touch the mesh/fitting stage.
Reports all three grid sizes' final transitions side-by-side for
direct comparison.

Run: python -m tryon.grid_size_6_visual_check
"""

import io
import os
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw

from tryon.correspondence_shoulders_test import find_garment_underarms
from tryon.garment_shoulder_region_extraction import (
    local_dominant_orientation, angle_diff, find_sustained_transition, get_binary_mask
)

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
PLAIN_SHIRT_ID = "7e16734e-6cb2-4254-adf5-4695549ad5db"
OUTPUT_DIR = "outputs"
DEVIATION_THRESHOLD_DEG = 20
REFERENCE_Y_BELOW_UA = 60


def get_reference_orientation_g(gray, mask, underarm_y, gw, grid_size):
    y_start = underarm_y + REFERENCE_Y_BELOW_UA
    y_end = y_start + 40
    angles = []
    for y in range(y_start, y_end, grid_size):
        for x in range(0, gw, grid_size):
            patch_mask = mask[y:y+grid_size, x:x+grid_size]
            if patch_mask.sum() < (grid_size*grid_size*0.5):
                continue
            gray_patch = gray[y:y+grid_size, x:x+grid_size]
            angle, coherence = local_dominant_orientation(gray_patch)
            if angle is not None and coherence >= 0.3:
                angles.append(angle)
    return np.median(angles) if angles else None


def walk_side_g(gray, mask, underarm_pt, reference_angle, gw, side, grid_size):
    ua_x, ua_y = underarm_pt
    results = []
    for y in range(ua_y, 0, -grid_size):
        row_mask = mask[y]
        xs = np.where(row_mask)[0]
        if len(xs) == 0:
            continue
        outer_x = xs.min() if side == "left" else xs.max()
        x_start = max(0, outer_x - grid_size) if side == "left" else outer_x
        x_end = outer_x if side == "left" else min(gw, outer_x + grid_size)

        patch_mask = mask[y:y+grid_size, x_start:x_end]
        gray_patch = gray[y:y+grid_size, x_start:x_end]
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
    print(f"Underarms: L={left_ua}  R={right_ua}\n")
    underarm_y = max(left_ua[1], right_ua[1])

    print(f"{'GRID_SIZE':<12}{'LEFT transition':<40}{'RIGHT transition'}")
    print("-" * 90)
    transitions_by_size = {}
    for grid_size in [12, 8, 6]:
        reference_angle = get_reference_orientation_g(gray, garment_mask, underarm_y, gw, grid_size)
        left_walk = walk_side_g(gray, garment_mask, left_ua, reference_angle, gw, "left", grid_size)
        right_walk = walk_side_g(gray, garment_mask, right_ua, reference_angle, gw, "right", grid_size)
        left_t = find_sustained_transition(left_walk)
        right_t = find_sustained_transition(right_walk)
        transitions_by_size[grid_size] = (left_walk, right_walk, left_t, right_t)
        left_str = f"{left_t}" if left_t else "None"
        right_str = f"{right_t}" if right_t else "None"
        print(f"{grid_size:<12}{left_str:<40}{right_str}")

    # Visualize GRID_SIZE=6 specifically — every walk point, transition highlighted
    left_walk, right_walk, left_t, right_t = transitions_by_size[6]

    canvas = garment_img.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    for pt, color, label in [(left_ua, (0,150,255), "L_UA"), (right_ua, (0,150,255), "R_UA")]:
        x, y = pt
        draw.ellipse([x-6,y-6,x+6,y+6], outline=color, width=3)
        draw.text((x+8,y-6), label, fill=color)

    for walk in [left_walk, right_walk]:
        for y, x, angle, deviates in walk:
            color = (255,80,80) if deviates is True else (0,200,0) if deviates is False else (150,150,150)
            draw.ellipse([x-4,y-4,x+4,y+4], outline=color, width=2)

    # Highlight the actual returned transition points distinctly
    for t, color, label in [(left_t, (255,255,0), "L_shoulder(returned)"), (right_t, (255,0,255), "R_shoulder(returned)")]:
        if t is None:
            continue
        y, x, angle, deviates = t
        draw.ellipse([x-9,y-9,x+9,y+9], outline=color, width=4)
        draw.text((x+11,y-9), label, fill=color)

    canvas.save(f"{OUTPUT_DIR}/grid_size_6_visual_check.png")
    print(f"\nSaved: {OUTPUT_DIR}/grid_size_6_visual_check.png")
    print(f"\nLegend: red=deviates, green=matches, gray=skipped, "
          f"yellow/magenta ring=actual returned transition point (GRID_SIZE=6)")


if __name__ == "__main__":
    main()