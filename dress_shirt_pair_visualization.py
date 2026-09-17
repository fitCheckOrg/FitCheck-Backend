"""
Visual identification of the three unexplained dress-shirt points.
No selector logic, no merge rule, no production changes — pure
visualization for direct inspection.

Run: python dress_shirt_pair_visualization.py
"""

import io
import requests
from PIL import Image, ImageDraw

ITEM_ID = "6a53c00f-4f84-4e64-ad3b-5d2753504481"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"

P1 = (289, 677)  # 99.9°
P2 = (307, 669)  # 35.3°
P3 = (356, 676)  # 98.3°


def main():
    result_data = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{ITEM_ID}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    image_bytes = requests.get(result_data["clean_image_url"]).content
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Full image with all three marked
    full = image.copy()
    draw = ImageDraw.Draw(full)
    for pt, color, label in [(P1, (255,0,0), "P1 (99.9°)"),
                                (P2, (0,255,0), "P2 (35.3°)"),
                                (P3, (0,150,255), "P3 (98.3°)")]:
        x, y = pt
        draw.ellipse([x-8,y-8,x+8,y+8], outline=color, width=3)
        draw.text((x+10,y-8), label, fill=color)
    full.save("dress_shirt_pairs_full.png")

    # Zoomed crop around the three points
    xs = [P1[0], P2[0], P3[0]]
    ys = [P1[1], P2[1], P3[1]]
    pad = 60
    box = (min(xs)-pad, min(ys)-pad, max(xs)+pad, max(ys)+pad)
    zoom = image.crop(box)
    zoom = zoom.resize((zoom.width*3, zoom.height*3), Image.LANCZOS)

    draw_zoom = ImageDraw.Draw(zoom)
    scale = 3
    for pt, color, label in [(P1, (255,0,0), "P1"), (P2, (0,255,0), "P2"), (P3, (0,150,255), "P3")]:
        zx = (pt[0] - box[0]) * scale
        zy = (pt[1] - box[1]) * scale
        draw_zoom.ellipse([zx-10,zy-10,zx+10,zy+10], outline=color, width=3)
        draw_zoom.text((zx+12,zy-10), label, fill=color)
    zoom.save("dress_shirt_pairs_zoom.png")

    print("Saved: dress_shirt_pairs_full.png")
    print("Saved: dress_shirt_pairs_zoom.png")


if __name__ == "__main__":
    main()