"""
Isolates ONE variable from the H1 fitting experiment: is the 
derived avatar underarm point (t=0.13) actually placed correctly 
on THIS specific avatar photo, independent of any garment overlay?

No garment involved at all — just the avatar photo + the derived 
point, so we can see exactly what the fitting code believes, 
compared to the visible real armpit.

Run: python verify_avatar_underarm_point.py
"""

import io
import requests
from PIL import Image, ImageDraw

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
UNDERARM_T = 0.13


def main():
    avatar_resp = requests.get(
        f"http://127.0.0.1:8000/api/avatar/me?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    img_resp = requests.get(avatar_resp["processed_photo_url"])
    avatar_img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
    w, h = avatar_img.size

    keypoints = avatar_resp["pose_keypoints"]

    def to_px(pt):
        return (pt["x"] * w, pt["y"] * h)

    left_shoulder = to_px(keypoints["left_shoulder"])
    right_shoulder = to_px(keypoints["right_shoulder"])
    left_hip = to_px(keypoints["left_hip"])
    right_hip = to_px(keypoints["right_hip"])

    def interpolate(s, hp, t):
        return (s[0] + t * (hp[0] - s[0]), s[1] + t * (hp[1] - s[1]))

    left_underarm = interpolate(left_shoulder, left_hip, UNDERARM_T)
    right_underarm = interpolate(right_shoulder, right_hip, UNDERARM_T)

    canvas = avatar_img.copy()
    draw = ImageDraw.Draw(canvas)

    def mark(pt, color, label, r=8):
        x, y = pt
        draw.ellipse([x-r, y-r, x+r, y+r], outline=color, width=3)
        draw.text((x+r+3, y-6), label, fill=color)

    mark(left_shoulder, (0, 255, 0), "L_shoulder", r=6)
    mark(right_shoulder, (0, 255, 0), "R_shoulder", r=6)
    mark(left_hip, (255, 165, 0), "L_hip", r=6)
    mark(right_hip, (255, 165, 0), "R_hip", r=6)
    mark(left_underarm, (255, 0, 0), "derived_L_underarm (t=0.13)")
    mark(right_underarm, (0, 200, 255), "derived_R_underarm (t=0.13)")

    canvas.save("verify_avatar_underarm_point.png")
    print(f"left_underarm derived at: ({left_underarm[0]:.1f}, {left_underarm[1]:.1f})")
    print(f"right_underarm derived at: ({right_underarm[0]:.1f}, {right_underarm[1]:.1f})")
    print(f"Saved: verify_avatar_underarm_point.png")
    print(f"\n>>> Does the red/cyan dot land on the ACTUAL visible armpit, or above/below it?")


if __name__ == "__main__":
    main()