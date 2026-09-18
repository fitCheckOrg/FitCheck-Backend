"""
Full convexity-defect enumeration for the new rugby-shirt failure
(item 4 from the fresh batch), same rigor as ba2e5aa1's original
diagnosis. Identifies the item by matching the truncated filename
prefix against the live wardrobe, then enumerates every defect.

Run: python item4_defect_enumeration.py
"""

import io
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw

ALPHA_THRESHOLD = 10
MIN_DEFECT_DEPTH = 1000
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
TARGET_ID_PREFIX = "729ddf0c"  # from the saved filename layer0_batch_729ddf0c.png


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255, image


def main():
    result = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/?user_id={AVATAR_USER_ID}"
    ).json()["data"]

    target_item = None
    for item in result.get("items", []):
        if item["item_id"].startswith(TARGET_ID_PREFIX):
            target_item = item
            break

    if not target_item:
        print(f"Could not find item with ID prefix {TARGET_ID_PREFIX}")
        return

    print(f"Found: {target_item['item_id']} — {target_item.get('item_type', 'unknown')}\n")

    image_bytes = requests.get(target_item["clean_image_url"]).content
    binary_mask, original = get_binary_mask(image_bytes)
    h_img, w_img = binary_mask.shape

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)

    hull_indices = cv2.convexHull(main_contour, returnPoints=False)
    defects = cv2.convexityDefects(main_contour, hull_indices)

    if defects is None:
        print("No convexity defects found at all.")
        return

    defects_flat = defects.reshape(-1, 4)
    all_candidates = []
    for i in range(len(defects_flat)):
        _, _, far_idx, depth = defects_flat[i]
        far_idx, depth = int(far_idx), int(depth)
        far_pt = main_contour[far_idx][0]
        x, y = int(far_pt[0]), int(far_pt[1])
        x_pct = x / w_img * 100
        y_pct = y / h_img * 100
        side = "LEFT" if x_pct < 50 else "RIGHT"
        all_candidates.append({
            "depth": depth, "x": x, "y": y,
            "x_pct": x_pct, "y_pct": y_pct, "side": side
        })

    all_candidates.sort(key=lambda c: -c["depth"])

    print(f"Image dimensions: {w_img} x {h_img}")
    print(f"Total convexity defects found: {len(all_candidates)}")
    print(f"Current MIN_DEFECT_DEPTH threshold: {MIN_DEFECT_DEPTH}\n")

    print(f"{'rank':<6}{'depth':<10}{'(x,y)':<16}{'x%':<8}{'y%':<8}{'side':<8}{'above threshold?'}")
    print("-" * 70)
    for rank, c in enumerate(all_candidates, 1):
        above = "YES" if c["depth"] > MIN_DEFECT_DEPTH else "no"
        coord_str = f"({c['x']},{c['y']})"
        print(f"{rank:<6}{c['depth']:<10}{coord_str:<16}"
              f"{c['x_pct']:<8.1f}{c['y_pct']:<8.1f}{c['side']:<8}{above}")

    left_side = [c for c in all_candidates if c["side"] == "LEFT"]
    print(f"\nAll LEFT-side defects, by depth:")
    for c in left_side:
        print(f"  depth={c['depth']}  pos=({c['x']},{c['y']})  ({c['x_pct']:.1f}%, {c['y_pct']:.1f}%)")

    canvas = original.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    pts = [(int(p[0][0]), int(p[0][1])) for p in main_contour]
    draw.line(pts + [pts[0]], fill=(80, 80, 80), width=1)
    for rank, c in enumerate(all_candidates, 1):
        color = (255, 0, 0) if c["depth"] > MIN_DEFECT_DEPTH else (100, 100, 255)
        x, y = c["x"], c["y"]
        draw.ellipse([x-7, y-7, x+7, y+7], outline=color, width=2)
        draw.text((x+9, y-6), f"#{rank}", fill=color)
    canvas.save("item4_defects_enumerated.png")
    print(f"\nSaved: item4_defects_enumerated.png")


if __name__ == "__main__":
    main()