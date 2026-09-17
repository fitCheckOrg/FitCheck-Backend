"""
Isolated landmark validation — ONLY derive_avatar_geometry() and
raw hip pose_keypoints, marked directly on the current avatar
photo. No contour sampling, no warping, no interpolation, no new
constants.

Answers exactly one question: do L_UA/R_UA/L_HIP/R_HIP land on
their real anatomical locations on the CURRENT stored avatar image?

Run: python -m tryon.avatar_landmarks_diagnostic
"""

import io
import requests
from PIL import Image, ImageDraw

from workers.avatar.geometry_derivation import derive_avatar_geometry

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"


def main():
    import os
    os.makedirs("outputs", exist_ok=True)

    avatar_resp = requests.get(
        f"http://127.0.0.1:8000/api/avatar/me?user_id={AVATAR_USER_ID}"
    ).json()["data"]

    print(f"avatar_id: {avatar_resp.get('avatar_id')}")
    print(f"avatar_version: {avatar_resp.get('avatar_version')}")
    print(f"photo_width: {avatar_resp.get('photo_width')}  photo_height: {avatar_resp.get('photo_height')}")
    print(f"processed_photo_url: {avatar_resp.get('processed_photo_url')}")

    avatar_img_resp = requests.get(avatar_resp["processed_photo_url"])
    avatar_img = Image.open(io.BytesIO(avatar_img_resp.content)).convert("RGB")
    aw, ah = avatar_img.size
    print(f"Actual downloaded image dimensions: {aw} x {ah}")

    pose_keypoints = avatar_resp["pose_keypoints"]
    avatar_geometry = derive_avatar_geometry(pose_keypoints)

    l_ua = (avatar_geometry.left_underarm.x_pct / 100 * aw,
             avatar_geometry.left_underarm.y_pct / 100 * ah)
    r_ua = (avatar_geometry.right_underarm.x_pct / 100 * aw,
             avatar_geometry.right_underarm.y_pct / 100 * ah)
    l_hip = (pose_keypoints["left_hip"]["x"] * aw, pose_keypoints["left_hip"]["y"] * ah)
    r_hip = (pose_keypoints["right_hip"]["x"] * aw, pose_keypoints["right_hip"]["y"] * ah)

    print(f"\nL_UA:  {l_ua}")
    print(f"R_UA:  {r_ua}")
    print(f"L_HIP: {l_hip}")
    print(f"R_HIP: {r_hip}")

    canvas = avatar_img.copy()
    draw = ImageDraw.Draw(canvas)

    for pt, color, label in [
        (l_ua, (0, 255, 0), "L_UA"),
        (r_ua, (0, 150, 255), "R_UA"),
        (l_hip, (255, 165, 0), "L_HIP"),
        (r_hip, (255, 0, 255), "R_HIP"),
    ]:
        x, y = pt
        draw.ellipse([x-8, y-8, x+8, y+8], outline=color, width=3)
        draw.text((x+10, y-8), label, fill=color)

    canvas.save("outputs/avatar_landmarks_diagnostic.png")

    import hashlib
    with open("outputs/avatar_landmarks_diagnostic.png", "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    print(f"\nSaved: outputs/avatar_landmarks_diagnostic.png")
    print(f"FILE MD5 HASH: {file_hash}")


if __name__ == "__main__":
    main()