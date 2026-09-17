"""
Checks whether the RGB differences seen in seam_diagonal_check are
explained by real stripe-crossing (a confound) rather than a
transform mismatch. Samples the ORIGINAL garment at pairs of points
a few pixels apart, comparable to the offset used in the seam check.

Run: python -m tryon.stripe_confound_check
"""

import io
import requests
import numpy as np
from PIL import Image

GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"


def main():
    garment_resp = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{GOLDEN_GARMENT_ID}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    garment_bytes = requests.get(garment_resp["clean_image_url"]).content
    garment_img = np.array(Image.open(io.BytesIO(garment_bytes)).convert("RGB"))

    # Sample near the real garment underarm line, testing vertical 
    # offsets of similar magnitude to what the seam check used
    test_points = [(150, 200), (150, 250), (200, 230), (250, 260), (300, 210)]
    OFFSET_Y = 6  # comparable total separation to the 3px+3px seam-check offset

    print(f"{'base_point':<14}{'rgb_at_point':<18}{'rgb_6px_below':<18}{'diff'}")
    print("-" * 60)
    for x, y in test_points:
        rgb1 = garment_img[y, x]
        rgb2 = garment_img[y + OFFSET_Y, x]
        diff = np.linalg.norm(rgb1.astype(int) - rgb2.astype(int))
        print(f"({x},{y})       {tuple(rgb1)}          {tuple(rgb2)}          {diff:.1f}")

    print(f"\n>>> If these diffs are ALSO large/inconsistent (similar magnitude "
          f"to the 2.8-45.8 range seen in the seam check), that confirms "
          f"striped fabric naturally produces this much local variation — "
          f"the seam RGB diffs are a texture confound, not a transform "
          f"mismatch.")


if __name__ == "__main__":
    main()