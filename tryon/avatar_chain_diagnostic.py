"""
Diagnostic ONLY — no warp, no composite. Draws the avatar contour,
underarms, hip references, and both sampled chains for direct
visual verification before trusting them in any warp.

Run: python -m tryon.avatar_chain_diagnostic
"""

import io
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw

from tryon.avatar_torso_sampler import sample_torso_side
from workers.avatar.geometry_derivation import derive_avatar_geometry

ALPHA_THRESHOLD = 10
NUM_SAMPLES = 5
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255, image


def main():
    avatar_resp = requests.get(
        f"http://127.0.0.1:8000/api/avatar/me?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    avatar_img_resp = requests.get(avatar_resp["processed_photo_url"])
    avatar_bytes = avatar_img_resp.content
    binary_mask, original = get_binary_mask(avatar_bytes)
    aw, ah = original.size

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)

    pose_keypoints = avatar_resp["pose_keypoints"]
    avatar_geometry = derive_avatar_geometry(pose_keypoints)

    a_left_ua = (avatar_geometry.left_underarm.x_pct / 100 * aw,
                  avatar_geometry.left_underarm.y_pct / 100 * ah)
    a_right_ua = (avatar_geometry.right_underarm.x_pct / 100 * aw,
                   avatar_geometry.right_underarm.y_pct / 100 * ah)
    a_left_hip = (pose_keypoints["left_hip"]["x"] * aw, pose_keypoints["left_hip"]["y"] * ah)
    a_right_hip = (pose_keypoints["right_hip"]["x"] * aw, pose_keypoints["right_hip"]["y"] * ah)

    left_chain = sample_torso_side(main_contour, a_left_ua, a_left_hip, is_left_side=True, num_samples=NUM_SAMPLES)
    right_chain = sample_torso_side(main_contour, a_right_ua, a_right_hip, is_left_side=False, num_samples=NUM_SAMPLES)

    print(f"Left underarm: {a_left_ua}  Left hip: {a_left_hip}")
    print(f"Right underarm: {a_right_ua}  Right hip: {a_right_hip}")
    print(f"Left chain: {left_chain}")
    print(f"Right chain: {right_chain}")

    canvas = original.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    pts = [(int(p[0][0]), int(p[0][1])) for p in main_contour]
    draw.line(pts + [pts[0]], fill=(150, 150, 150), width=1)

    for pt, color, label in [(a_left_ua, (0,255,0), "L_UA"), (a_right_ua, (0,255,0), "R_UA"),
                                (a_left_hip, (255,165,0), "L_hip"), (a_right_hip, (255,165,0), "R_hip")]:
        x, y = pt
        draw.ellipse([x-6,y-6,x+6,y+6], outline=color, width=2)
        draw.text((x+8,y-6), label, fill=color)

    for i, pt in enumerate(left_chain):
        x, y = pt
        draw.ellipse([x-6,y-6,x+6,y+6], outline=(255,0,0), width=3)
        draw.text((x+8,y), f"L{i}", fill=(255,0,0))
    for i, pt in enumerate(right_chain):
        x, y = pt
        draw.ellipse([x-6,y-6,x+6,y+6], outline=(0,150,255), width=3)
        draw.text((x+8,y), f"R{i}", fill=(0,150,255))

    canvas.save("avatar_chain_diagnostic.png")
    print("\nSaved: avatar_chain_diagnostic.png")


if __name__ == "__main__":
    main()