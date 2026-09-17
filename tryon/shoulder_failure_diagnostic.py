"""
Visual diagnostic for the two garments that failed shoulder
detection in the generalization test. Draws the full contour,
underarms, and every candidate point checked during the walk (color-
coded True/False/None), so we can see directly why no sustained
transition was found.

Run: python -m tryon.shoulder_failure_diagnostic
"""

import io
import os
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw

from tryon.correspondence_shoulders_test import find_garment_underarms
from tryon.garment_shoulder_region_extraction import (
    get_reference_orientation, walk_side, get_binary_mask
)

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
OUTPUT_DIR = "outputs"

FAILING_GARMENTS = [
    {"item_id": "7aeb558a-82d0-4d24-a501-ae55377d130b", "label": "fresh_5"},
    {"item_id": "d7256fcb-c37d-4a44-88de-d593ad61f20e", "label": "fresh_6"},
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for g in FAILING_GARMENTS:
        print(f"\n{'='*60}\n{g['label']} ({g['item_id']})\n{'='*60}")

        garment_resp = requests.get(
            f"http://127.0.0.1:8000/api/wardrobe/{g['item_id']}?user_id={AVATAR_USER_ID}"
        ).json()["data"]
        garment_bytes = requests.get(garment_resp["clean_image_url"]).content
        garment_mask, garment_img = get_binary_mask(garment_bytes)
        gw, gh = garment_img.size
        gray = cv2.cvtColor(np.array(garment_img.convert("RGB")), cv2.COLOR_RGB2GRAY)

        left_ua, right_ua = find_garment_underarms(garment_mask, gw)
        print(f"Underarms: L={left_ua}  R={right_ua}")

        underarm_y = max(left_ua[1], right_ua[1])
        reference_angle = get_reference_orientation(gray, garment_mask, underarm_y, gw)
        print(f"Reference torso orientation: {reference_angle:.1f}°")

        left_walk = walk_side(gray, garment_mask, left_ua, reference_angle, gw, "left")
        right_walk = walk_side(gray, garment_mask, right_ua, reference_angle, gw, "right")

        canvas = garment_img.convert("RGB").copy()
        draw = ImageDraw.Draw(canvas)

        for walk in [left_walk, right_walk]:
            for y, x, angle, deviates in walk:
                if deviates is True:
                    color = (255, 0, 0)
                elif deviates is False:
                    color = (0, 200, 0)
                else:
                    color = (150, 150, 150)
                draw.ellipse([x-4, y-4, x+4, y+4], outline=color, width=2)

        for pt, label in [(left_ua, "L_UA"), (right_ua, "R_UA")]:
            x, y = pt
            draw.ellipse([x-6, y-6, x+6, y+6], outline=(0,150,255), width=3)
            draw.text((x+8, y-6), label, fill=(0,150,255))

        out_path = f"{OUTPUT_DIR}/shoulder_failure_{g['label']}.png"
        canvas.save(out_path)
        print(f"Saved: {out_path}")

    print(f"\n>>> Red = deviates (sleeve-like), Green = matches torso, "
          f"Gray = skipped (low coherence, no confident orientation). "
          f"Look for: is there simply no red signal at all (confirms "
          f"texture-dependency), or does red appear but never 3-in-a-row "
          f"(a different, threshold-related cause)?")


if __name__ == "__main__":
    main()