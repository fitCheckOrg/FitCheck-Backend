"""
Diagnoses WHY the plain white t-shirt fails shoulder detection —
specifically: is there zero usable orientation signal at all
(confirms plain fabric has nothing for this method to detect), or
is there weak-but-present signal that's just below the current
coherence threshold (a tuning question, not a fundamental limit)?

Run: python -m tryon.plain_shirt_diagnostic
"""

import io
import os
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw

from tryon.correspondence_shoulders_test import find_garment_underarms
from tryon.garment_shoulder_region_extraction import (
    get_reference_orientation, walk_side, local_dominant_orientation, get_binary_mask
)

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
OUTPUT_DIR = "outputs"
PLAIN_SHIRT_ID = "7e16734e-6cb2-4254-adf5-4695549ad5db"
GRID_SIZE = 12


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
    reference_angle = get_reference_orientation(gray, garment_mask, underarm_y, gw)
    print(f"Reference torso orientation: {reference_angle}")

    left_walk = walk_side(gray, garment_mask, left_ua, reference_angle, gw, "left")
    right_walk = walk_side(gray, garment_mask, right_ua, reference_angle, gw, "right")

    print(f"\n--- LEFT walk (raw, all rows) ---")
    for y, x, angle, deviates in left_walk:
        angle_str = f"{angle:.1f}°" if angle is not None else "N/A"
        print(f"  y={y:4d}  x={x:4d}  angle={angle_str:>7}  {deviates}")

    print(f"\n--- RIGHT walk (raw, all rows) ---")
    for y, x, angle, deviates in right_walk:
        angle_str = f"{angle:.1f}°" if angle is not None else "N/A"
        print(f"  y={y:4d}  x={x:4d}  angle={angle_str:>7}  {deviates}")

    # Directly measure coherence (not just angle) at every sampled 
    # patch, to see if signal exists but is weak vs. truly absent
    print(f"\n--- Raw coherence values along both walks ---")
    for side_name, walk, underarm_pt in [("LEFT", left_walk, left_ua), ("RIGHT", right_walk, right_ua)]:
        print(f"\n{side_name}:")
        for y, x, angle, deviates in walk:
            if side_name == "LEFT":
                x_start, x_end = max(0, x-GRID_SIZE), x
            else:
                x_start, x_end = x, min(gw, x+GRID_SIZE)
            patch = gray[y:y+GRID_SIZE, x_start:x_end]
            _, coherence = local_dominant_orientation(patch)
            print(f"  y={y}  coherence={coherence:.3f}  (threshold is 0.25)")

    # Visualize: full contour + underarms + every walk point, colored 
    # by coherence strength directly (not just True/False/None)
    canvas = garment_img.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for pt, color, label in [(left_ua, (0,150,255), "L_UA"), (right_ua, (0,150,255), "R_UA")]:
        x, y = pt
        draw.ellipse([x-6,y-6,x+6,y+6], outline=color, width=3)
        draw.text((x+8,y-6), label, fill=color)
    for walk in [left_walk, right_walk]:
        for y, x, angle, deviates in walk:
            color = (255,0,0) if deviates is True else (0,200,0) if deviates is False else (150,150,150)
            draw.ellipse([x-4,y-4,x+4,y+4], outline=color, width=2)

    canvas.save(f"{OUTPUT_DIR}/plain_shirt_diagnostic.png")
    print(f"\nSaved: {OUTPUT_DIR}/plain_shirt_diagnostic.png")
    print(f"\n>>> DECISION: if coherence values are consistently LOW (near or "
          f"below 0.25) throughout, that confirms genuinely absent signal — "
          f"plain fabric has no directional structure to detect, and we "
          f"need a different signal source entirely. If coherence is "
          f"moderate/borderline, this is closer to a threshold-tuning issue.")


if __name__ == "__main__":
    main()