"""
Layer 0 Investigation — Diagnostic 2: batch convexity-defect
enumeration across several fresh garments.

Frozen detector, unchanged. No threshold tuning. For each new
garment, reports full defect population (not just the winning
left/right pair) so we can determine whether ba2e5aa1's failure
was a one-off or a repeatable class.

Run: python layer0_batch_enumeration.py
"""

import io
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw

ALPHA_THRESHOLD = 10
MIN_DEFECT_DEPTH = 1000  # current production value, unchanged — 
                            # just used as a reference marker in output

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"

# Every item_id used anywhere in this entire investigation, excluded
DEVELOPMENT_ITEM_IDS = {
    "ac7fffb7-2e3a-40f8-894a-0e85fc245373",
    "25989827-db9d-4d65-90b3-12f5357b0b44",
    "2ea4bb57-a358-4349-8aaf-204ded0772ab",
    "6a53c00f-4f84-4e64-ad3b-5d2753504481",
    "bb7763f6-f1df-4b26-9522-94299149bb5e",
    "f2944c50-5757-4abc-81a6-c1380f8419a0",
    "d20577a1-01e9-4d6f-87a1-5ca89c3badba",
    "52f5fd29-d817-49bb-a0c8-ba181ba47157",
    "cb13467e-585f-4a4f-82fe-bac29082424e",
    "ded59bd3-57a1-43b1-888a-867e4362d2d8",
    "c93a3138-f40f-4e91-8d91-be3ff1c3065f",
    "5d19a29d-a7fa-4579-a1ba-2e374c3fd410",
    "fe419f35-3b62-43b9-842c-8883651e9c4f",
    "62930780-680d-4653-a8fa-daa7a6900617",
    "bd85602f-8c93-4d2c-99e9-7eafdb6da425",
    "2a32b5bb-c3ea-48f9-ac8d-5849eee866ba",
    "e398e450-a4be-447a-b981-5a57297c7e66",  # turtleneck — discarded, not real
    "ba2e5aa1-bfb5-462f-aaad-fe378f658c42",  # already fully characterized
}


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255, image


def enumerate_defects(main_contour, w_img, h_img):
    hull_indices = cv2.convexHull(main_contour, returnPoints=False)
    try:
        defects = cv2.convexityDefects(main_contour, hull_indices)
    except cv2.error:
        return []
    if defects is None:
        return []

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
    return all_candidates


def main():
    result = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/?user_id={AVATAR_USER_ID}"
    ).json()["data"]

    all_items = result.get("items", [])
    fresh_items = [
        item for item in all_items
        if item["item_id"] not in DEVELOPMENT_ITEM_IDS
        and item.get("category") == "top"
        and not item.get("is_archived", False)
    ]

    print(f"Found {len(fresh_items)} fresh items for Layer 0 batch\n")

    for item in fresh_items:
        print(f"\n{'='*75}")
        print(f"{item['item_id']} — {item.get('item_type', 'unknown')}")
        print('='*75)

        image_bytes = requests.get(item["clean_image_url"]).content
        binary_mask, original = get_binary_mask(image_bytes)
        h_img, w_img = binary_mask.shape

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_contour = max(contours, key=cv2.contourArea)

        defects = enumerate_defects(main_contour, w_img, h_img)
        print(f"Total defects: {len(defects)}")

        left_side = [d for d in defects if d["side"] == "LEFT"]
        right_side = [d for d in defects if d["side"] == "RIGHT"]

        left_winner = left_side[0] if left_side else None
        right_winner = right_side[0] if right_side else None

        if left_winner:
            print(f"LEFT winner:  depth={left_winner['depth']:<8} "
                  f"pos=({left_winner['x']},{left_winner['y']}) "
                  f"({left_winner['x_pct']:.1f}%, {left_winner['y_pct']:.1f}%)")
        if right_winner:
            print(f"RIGHT winner: depth={right_winner['depth']:<8} "
                  f"pos=({right_winner['x']},{right_winner['y']}) "
                  f"({right_winner['x_pct']:.1f}%, {right_winner['y_pct']:.1f}%)")

        # Flag if the winning y-position looks suspiciously high (collar-zone-like)
        for side_name, winner in [("LEFT", left_winner), ("RIGHT", right_winner)]:
            if winner and winner["y_pct"] < 20:
                print(f"  ⚠ {side_name} winner sits at y={winner['y_pct']:.1f}% — "
                      f"suspiciously high, possible collar/neckline substitution "
                      f"(same failure pattern as ba2e5aa1)")

        canvas = original.convert("RGB").copy()
        draw = ImageDraw.Draw(canvas)
        pts = [(int(p[0][0]), int(p[0][1])) for p in main_contour]
        draw.line(pts + [pts[0]], fill=(80, 80, 80), width=1)

        if left_winner:
            x, y = left_winner["x"], left_winner["y"]
            draw.ellipse([x-8,y-8,x+8,y+8], outline=(255,0,0), width=3)
            draw.text((x+10,y-8), "L_underarm", fill=(255,0,0))
        if right_winner:
            x, y = right_winner["x"], right_winner["y"]
            draw.ellipse([x-8,y-8,x+8,y+8], outline=(0,150,255), width=3)
            draw.text((x+10,y-8), "R_underarm", fill=(0,150,255))

        filename = f"layer0_batch_{item['item_id'][:8]}.png"
        canvas.save(filename)
        print(f"Saved: {filename}")
        print(f"\n>>> RECORD: sleeve type? correct/incorrect? if incorrect, what won instead?")


if __name__ == "__main__":
    main()