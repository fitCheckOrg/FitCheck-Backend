"""
Visual identification of hanging-shirt RIGHT's ACTUAL candidate
pair — (442,447) and (366,458) — correcting the earlier mislabeled
check, which turned out to show LEFT's own pair instead.

Run: python hanging_shirt_right_actual_visualization.py
"""

import io
import requests
from PIL import Image, ImageDraw

ITEM_ID = "25989827-db9d-4d65-90b3-12f5357b0b44"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"

CONFIRMED_CUFF = (442, 447)  # 81.5°
COMPETING = (366, 458)        # 89.7° — higher curvature than confirmed


def main():
    result_data = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{ITEM_ID}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    image_bytes = requests.get(result_data["clean_image_url"]).content
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    full = image.copy()
    draw = ImageDraw.Draw(full)
    for pt, color, label in [(CONFIRMED_CUFF, (255,0,0), "CONFIRMED (81.5°)"),
                                (COMPETING, (0,255,0), "COMPETING (89.7°)")]:
        x, y = pt
        draw.ellipse([x-8,y-8,x+8,y+8], outline=color, width=3)
        draw.text((x+10,y-8), label, fill=color)
    full.save("hanging_shirt_right_actual_full.png")

    xs = [CONFIRMED_CUFF[0], COMPETING[0]]
    ys = [CONFIRMED_CUFF[1], COMPETING[1]]
    pad = 60
    box = (min(xs)-pad, min(ys)-pad, max(xs)+pad, max(ys)+pad)
    zoom = image.crop(box)
    scale = 4
    zoom = zoom.resize((zoom.width*scale, zoom.height*scale), Image.LANCZOS)

    draw_zoom = ImageDraw.Draw(zoom)
    for pt, color, label in [(CONFIRMED_CUFF, (255,0,0), "CONFIRMED"), (COMPETING, (0,255,0), "COMPETING")]:
        zx = (pt[0] - box[0]) * scale
        zy = (pt[1] - box[1]) * scale
        draw_zoom.ellipse([zx-10,zy-10,zx+10,zy+10], outline=color, width=3)
        draw_zoom.text((zx+12,zy-10), label, fill=color)
    zoom.save("hanging_shirt_right_actual_zoom.png")

    print("Saved: hanging_shirt_right_actual_full.png")
    print("Saved: hanging_shirt_right_actual_zoom.png")


if __name__ == "__main__":
    main()