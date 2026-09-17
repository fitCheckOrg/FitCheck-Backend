"""
Tests whether GRID_SIZE=12 is genuinely undersampling the left-side
transition, or whether the left side simply lacks the same
orientation discontinuity the right side has. Same algorithm, same
reference angle logic, same sustained-transition check — only
GRID_SIZE varies. No production threshold changes.

Run: python -m tryon.grid_size_resolution_test
"""

import io
import os
import requests
import numpy as np
import cv2
from PIL import Image

from tryon.correspondence_shoulders_test import find_garment_underarms
from tryon.garment_shoulder_region_extraction import (
    local_dominant_orientation, angle_diff, find_sustained_transition, get_binary_mask
)

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
PLAIN_SHIRT_ID = "7e16734e-6cb2-4254-adf5-4695549ad5db"
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

    for grid_size in [12, 8, 6]:
        print(f"{'='*70}\nGRID_SIZE = {grid_size}\n{'='*70}")

        reference_angle = get_reference_orientation_g(gray, garment_mask, underarm_y, gw, grid_size)
        print(f"Reference torso orientation: {reference_angle:.1f}°")

        left_walk = walk_side_g(gray, garment_mask, left_ua, reference_angle, gw, "left", grid_size)
        right_walk = walk_side_g(gray, garment_mask, right_ua, reference_angle, gw, "right", grid_size)

        print(f"\nLEFT walk ({len(left_walk)} samples):")
        for y, x, angle, deviates in left_walk:
            angle_str = f"{angle:.1f}°" if angle is not None else "N/A"
            print(f"  y={y:4d}  x={x:4d}  angle={angle_str:>7}  {deviates}")

        print(f"\nRIGHT walk ({len(right_walk)} samples):")
        for y, x, angle, deviates in right_walk:
            angle_str = f"{angle:.1f}°" if angle is not None else "N/A"
            print(f"  y={y:4d}  x={x:4d}  angle={angle_str:>7}  {deviates}")

        left_transition = find_sustained_transition(left_walk)
        right_transition = find_sustained_transition(right_walk)

        print(f"\nLEFT transition: {left_transition}")
        print(f"RIGHT transition: {right_transition}")
        print()


if __name__ == "__main__":
    main()