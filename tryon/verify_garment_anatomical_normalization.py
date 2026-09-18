"""
Task 1.2 mandatory gate. Tests whether swapping garment image-space
left/right labels produces genuine anatomical left/right, or silently
mirrors. Renders the golden garment with large, unambiguous
ANATOMICAL_LEFT / ANATOMICAL_RIGHT labels at the proposed-normalized
points, for direct human visual confirmation — no logo/brand
convention assumed or relied on.

The test a human applies: imagine wearing this garment, facing the
camera. Does ANATOMICAL_LEFT sit on YOUR left side (the garment's
right side as drawn, since you're facing the viewer)?

Run: python -m tryon.verify_garment_anatomical_normalization
"""

import requests
from PIL import Image, ImageDraw, ImageFont

from tryon.garment_shoulder_region_extraction import get_binary_mask
from tryon.correspondence_shoulders_test import find_garment_underarms
from tryon.piecewise_continuous_test import get_garment_shoulder_points

GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
OUTPUT_DIR = "outputs"


def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    garment_resp = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{GOLDEN_GARMENT_ID}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    garment_bytes = requests.get(garment_resp["clean_image_url"]).content
    mask, image = get_binary_mask(garment_bytes)
    w, h = image.size

    g_left_ua, g_right_ua = find_garment_underarms(mask, w)
    g_left_sh, g_right_sh = get_garment_shoulder_points(mask, image, g_left_ua, g_right_ua, w)

    print(f"Raw image-space results (small-x labeled 'left' by existing functions):")
    print(f"  underarm: small_x={g_left_ua}  large_x={g_right_ua}")
    print(f"  shoulder: small_x={g_left_sh}  large_x={g_right_sh}")

    # PROPOSED normalization: swap. small-x -> anatomical_right, 
    # large-x -> anatomical_left (mirrors the already-proven avatar rule)
    anatomical_right_underarm = g_left_ua   # was "small_x"
    anatomical_left_underarm = g_right_ua   # was "large_x"
    anatomical_right_shoulder = g_left_sh
    anatomical_left_shoulder = g_right_sh

    print(f"\nProposed anatomical labels (UNPROVEN — this is the hypothesis being tested):")
    print(f"  anatomical_left_underarm = {anatomical_left_underarm}")
    print(f"  anatomical_right_underarm = {anatomical_right_underarm}")
    print(f"  anatomical_left_shoulder = {anatomical_left_shoulder}")
    print(f"  anatomical_right_shoulder = {anatomical_right_shoulder}")

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    for pt, label, color in [
        (anatomical_left_underarm, "ANATOMICAL_LEFT (underarm)", (255, 0, 255)),
        (anatomical_right_underarm, "ANATOMICAL_RIGHT (underarm)", (0, 200, 255)),
        (anatomical_left_shoulder, "ANATOMICAL_LEFT (shoulder)", (255, 0, 255)),
        (anatomical_right_shoulder, "ANATOMICAL_RIGHT (shoulder)", (0, 200, 255)),
    ]:
        x, y = pt
        draw.ellipse([x-8, y-8, x+8, y+8], outline=color, width=4)
        draw.text((x+10, y-8), label, fill=color)

    out_path = f"{OUTPUT_DIR}/garment_anatomical_normalization_check.png"
    canvas.save(out_path)
    print(f"\nSaved: {out_path}")
    print(f"\n>>> HUMAN VERIFICATION REQUIRED: imagine wearing this garment, facing "
          f"the viewer. Does the magenta 'ANATOMICAL_LEFT' point sit on the side "
          f"that would actually be YOUR left arm/shoulder? If yes, the swap rule "
          f"is confirmed. If the labels feel backwards, the rule is wrong and "
          f"must not ship as written.")


if __name__ == "__main__":
    main()