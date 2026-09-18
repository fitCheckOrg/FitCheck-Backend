"""
Orientation regression gate for garment anatomical normalization.
Tests the proposed rule (small-x -> anatomical_right, large-x ->
anatomical_left) across all 7 confirmed in-contract garments from the
existing regression set — not just the golden garment.

This does NOT modify GarmentAsset serialization. Pure diagnostic.
Produces one image per garment for human visual confirmation, plus
a summary table.

Run: python -m tryon.verify_garment_orientation_regression
"""

import os
import requests
from PIL import Image, ImageDraw

from tryon.garment_shoulder_region_extraction import get_binary_mask
from tryon.correspondence_shoulders_test import find_garment_underarms
from tryon.piecewise_continuous_test import get_garment_shoulder_points

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
OUTPUT_DIR = "outputs"

# The 7 confirmed in-contract garments from the visual regression baseline
GARMENTS = [
    {"id": "ac7fffb7-2e3a-40f8-894a-0e85fc245373", "label": "golden_polo"},
    {"id": "5c22b2ea-fedc-4dbb-b649-fc645774d011", "label": "fresh_3_t-shirt"},
    {"id": "d5d05081-fdb8-4104-8b56-53bb06e2bc93", "label": "fresh_4_polo"},
    {"id": "3a46afa7-e179-4d8b-a007-ef9af08fe27b", "label": "fresh_5_short-sleeve"},
    {"id": "d7256fcb-c37d-4a44-88de-d593ad61f20e", "label": "fresh_7_polo"},
    {"id": "42b7a010-c4fa-4b8f-8ed2-10c3088967b5", "label": "fresh_8_polo"},
    {"id": "831098a3-8778-480f-9a11-aa9df826d0f2", "label": "fresh_9_polo"},
]


def render_orientation_check(garment_id, label):
    garment_resp = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{garment_id}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    garment_bytes = requests.get(garment_resp["clean_image_url"]).content
    mask, image = get_binary_mask(garment_bytes)
    w, h = image.size

    g_left_ua, g_right_ua = find_garment_underarms(mask, w)
    g_left_sh, g_right_sh = get_garment_shoulder_points(mask, image, g_left_ua, g_right_ua, w)

    # Same proposed rule as the golden-garment gate: small-x -> anatomical_right
    anatomical_right_underarm = g_left_ua if g_left_ua[0] < g_right_ua[0] else g_right_ua
    anatomical_left_underarm = g_right_ua if g_left_ua[0] < g_right_ua[0] else g_left_ua
    anatomical_right_shoulder = g_left_sh if g_left_sh[0] < g_right_sh[0] else g_right_sh
    anatomical_left_shoulder = g_right_sh if g_left_sh[0] < g_right_sh[0] else g_left_sh

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for pt, text, color in [
        (anatomical_left_underarm, "L", (255, 0, 255)),
        (anatomical_right_underarm, "R", (0, 200, 255)),
        (anatomical_left_shoulder, "L", (255, 0, 255)),
        (anatomical_right_shoulder, "R", (0, 200, 255)),
    ]:
        x, y = pt
        draw.ellipse([x-8, y-8, x+8, y+8], outline=color, width=4)
        draw.text((x+10, y-8), text, fill=color)

    out_path = f"{OUTPUT_DIR}/orientation_{label}.png"
    canvas.save(out_path)

    return {
        "label": label,
        "anatomical_left_underarm": anatomical_left_underarm,
        "anatomical_right_underarm": anatomical_right_underarm,
        "anatomical_left_shoulder": anatomical_left_shoulder,
        "anatomical_right_shoulder": anatomical_right_shoulder,
        "output_path": out_path,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"{'Garment':<24}{'L_underarm':<16}{'R_underarm':<16}{'output'}")
    print("-" * 90)

    results = []
    for g in GARMENTS:
        try:
            r = render_orientation_check(g["id"], g["label"])
            results.append(r)
            print(f"{r['label']:<24}{str(r['anatomical_left_underarm']):<16}"
                  f"{str(r['anatomical_right_underarm']):<16}{r['output_path']}")
        except Exception as e:
            print(f"{g['label']:<24}ERROR: {e}")

    print(f"\n>>> MANUAL GATE — inspect each saved image (magenta=proposed ANATOMICAL_LEFT, "
          f"cyan=proposed ANATOMICAL_RIGHT). For each garment, confirm: if worn facing "
          f"the viewer, does magenta actually land on the wearer's real left side?")
    print(f">>> Report PASS/FAIL per garment. A single FAIL means the small-x→right rule "
          f"is NOT a safe global normalization — do not lock it into GarmentAsset.")


if __name__ == "__main__":
    main()