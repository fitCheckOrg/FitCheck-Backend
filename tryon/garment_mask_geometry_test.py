"""
Garment mask geometry diagnostic — the analogous measurement to
torso_mask_geometry_test.py, applied to the golden garment's own
alpha mask. Same discipline: query real pixels row-by-row, no
correspondence to the avatar yet, no fitting, no scaling.

Run: python -m tryon.garment_mask_geometry_test
"""

import io
import os
import requests
import numpy as np
from PIL import Image, ImageDraw

GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
ALPHA_THRESHOLD = 10
OUTPUT_DIR = "outputs"


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD), image


def extract_row_boundaries(mask):
    h, w = mask.shape
    rows = []
    for y in range(h):
        row = mask[y]
        xs = np.where(row)[0]
        if len(xs) == 0:
            continue
        left = int(xs.min())
        right = int(xs.max())
        width = right - left
        rows.append((y, left, right, width))
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Fetching golden garment...")
    garment_resp = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{GOLDEN_GARMENT_ID}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    garment_bytes = requests.get(garment_resp["clean_image_url"]).content

    mask, garment_img = get_binary_mask(garment_bytes)
    gw, gh = garment_img.size
    print(f"Image size: {gw} x {gh}")
    print(f"Garment mask total pixels: {mask.sum()}")

    rows = extract_row_boundaries(mask)

    if not rows:
        print("\n>>> NO GARMENT PIXELS FOUND. Mask failure.")
        return

    top_y = rows[0][0]
    bottom_y = rows[-1][0]
    widths = [r[3] for r in rows]
    max_width = max(widths)
    max_width_y = rows[widths.index(max_width)][0]
    min_width = min(widths)
    min_width_y = rows[widths.index(min_width)][0]

    print(f"\nGarment bounds:")
    print(f"  top:    y={top_y}")
    print(f"  bottom: y={bottom_y}")
    print(f"  height: {bottom_y - top_y}px")

    print(f"\nWidth profile (every 15th row):")
    for r in rows[::15]:
        y, left, right, width = r
        print(f"  y={y:4d}  left={left:4d}  right={right:4d}  width={width:4d}")

    print(f"\nMaximum width: {max_width}px at y={max_width_y}")
    print(f"Minimum width: {min_width}px at y={min_width_y}")

    print(f"\nLargest row-to-row width jumps (top 5, for inspection only):")
    jumps = []
    for i in range(1, len(rows)):
        prev_width = rows[i-1][3]
        curr_width = rows[i][3]
        jump = abs(curr_width - prev_width)
        jumps.append((jump, rows[i][0], prev_width, curr_width))
    jumps.sort(reverse=True)
    for jump, y, prev_w, curr_w in jumps[:5]:
        print(f"  y={y}: width jumped from {prev_w} to {curr_w} (Δ{jump})")

    canvas = garment_img.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    for r in rows[::3]:
        y, left, right, width = r
        draw.point((left, y), fill=(255, 0, 0))
        draw.point((right, y), fill=(0, 150, 255))

    for y, label, color in [
        (top_y, "TOP", (0, 255, 0)),
        (bottom_y, "BOTTOM", (0, 255, 0)),
        (max_width_y, "MAX_WIDTH", (255, 255, 0)),
        (min_width_y, "MIN_WIDTH", (255, 0, 255)),
    ]:
        draw.line([(0, y), (gw, y)], fill=color, width=1)
        draw.text((5, y), label, fill=color)

    canvas.save(f"{OUTPUT_DIR}/garment_mask_geometry_test.png")
    print(f"\nSaved: {OUTPUT_DIR}/garment_mask_geometry_test.png")
    print(f"\n>>> DECISION GATE: does the garment's own width profile look "
          f"like a coherent shirt silhouette (narrow at collar, wide at "
          f"chest/underarm, tapering or steady toward hem), or does it "
          f"show unexpected discontinuities (e.g. sleeves merging into "
          f"torso width)?")


if __name__ == "__main__":
    main()