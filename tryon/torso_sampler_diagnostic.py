"""
Isolated diagnostic for sample_torso_side()'s first-sample bug.
Prints EVERY candidate found in the search corridor for one side,
so we can see directly why the closest point to the real underarm
wasn't selected.

Run: python -m tryon.torso_sampler_diagnostic
"""

import io
import os
import requests
import numpy as np
import cv2
from PIL import Image

from workers.avatar.geometry_derivation import derive_avatar_geometry

ALPHA_THRESHOLD = 10
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255


def diagnose_corridor(contour, underarm_point, hip_point, is_left_side, num_samples=5):
    ua_x, ua_y = underarm_point
    hip_x, hip_y = hip_point

    y_min, y_max = min(ua_y, hip_y), max(ua_y, hip_y)
    midline_x = (ua_x + hip_x) / 2
    corridor_half_width = abs(ua_x - hip_x) + 30

    print(f"\nunderarm={underarm_point}  hip={hip_point}")
    print(f"y_range=[{y_min:.1f}, {y_max:.1f}]  midline_x={midline_x:.1f}  "
          f"corridor_half_width={corridor_half_width:.1f}")
    print(f"corridor x-range: [{midline_x-corridor_half_width:.1f}, {midline_x+corridor_half_width:.1f}]")

    candidates = []
    for pt in contour:
        x, y = int(pt[0][0]), int(pt[0][1])
        if y_min <= y <= y_max and abs(x - midline_x) <= corridor_half_width:
            if is_left_side and x <= midline_x + corridor_half_width:
                candidates.append((x, y))
            elif not is_left_side and x >= midline_x - corridor_half_width:
                candidates.append((x, y))

    print(f"\nTotal candidates found in corridor: {len(candidates)}")

    # Show candidates closest to the TOP of the range (near underarm)
    candidates_sorted_by_y = sorted(candidates, key=lambda p: p[1])
    print(f"\nFirst 15 candidates, sorted by Y (closest to underarm height first):")
    for c in candidates_sorted_by_y[:15]:
        dist_to_ua = np.hypot(c[0]-ua_x, c[1]-ua_y)
        print(f"  pos={c}  dist_to_underarm={dist_to_ua:.1f}px")

    # What sample_torso_side actually picks for the FIRST target_y
    target_ys = np.linspace(y_min, y_max, num_samples)
    first_target_y = target_ys[0]
    closest = min(candidates, key=lambda p: abs(p[1] - first_target_y))
    print(f"\nfirst target_y = {first_target_y:.1f}")
    print(f"closest candidate selected: {closest}  "
          f"(actual dist_to_underarm={np.hypot(closest[0]-ua_x, closest[1]-ua_y):.1f}px)")


def main():
    avatar_resp = requests.get(
        f"http://127.0.0.1:8000/api/avatar/me?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    avatar_bytes = requests.get(avatar_resp["processed_photo_url"]).content
    avatar_img = Image.open(io.BytesIO(avatar_bytes)).convert("RGBA")
    aw, ah = avatar_img.size

    binary_mask = get_binary_mask(avatar_bytes)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)

    pose_keypoints = avatar_resp["pose_keypoints"]
    avatar_geometry = derive_avatar_geometry(pose_keypoints)

    l_ua = (avatar_geometry.left_underarm.x_pct / 100 * aw, avatar_geometry.left_underarm.y_pct / 100 * ah)
    r_ua = (avatar_geometry.right_underarm.x_pct / 100 * aw, avatar_geometry.right_underarm.y_pct / 100 * ah)
    l_hip = (pose_keypoints["left_hip"]["x"] * aw, pose_keypoints["left_hip"]["y"] * ah)
    r_hip = (pose_keypoints["right_hip"]["x"] * aw, pose_keypoints["right_hip"]["y"] * ah)

    print("="*70)
    print("LEFT SIDE")
    print("="*70)
    diagnose_corridor(main_contour, l_ua, l_hip, is_left_side=True)

    print("\n" + "="*70)
    print("RIGHT SIDE")
    print("="*70)
    diagnose_corridor(main_contour, r_ua, r_hip, is_left_side=False)


if __name__ == "__main__":
    main()