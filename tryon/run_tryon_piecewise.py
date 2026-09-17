"""
Piecewise-affine version of the vertical slice. Same golden
garment/avatar, same validated landmark detection — new warp
method only.

Fixed: order-agnostic destination-point offsetting (previously
assumed right_ua.x > left_ua.x, which is wrong for this avatar's
actual pose_keypoints layout).

Run: python -m tryon.run_tryon_piecewise
"""

import io
import os
import requests
import numpy as np
import cv2
from PIL import Image

from tryon.piecewise_warper import get_garment_hem_corners, piecewise_warp_garment
from workers.avatar.geometry_derivation import derive_avatar_geometry

ALPHA_THRESHOLD = 10
MIN_DEFECT_DEPTH = 1000

GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
OUTPUT_DIR = "outputs"


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255, image


def find_garment_underarms(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)
    w_img = binary_mask.shape[1]

    hull_indices = cv2.convexHull(main_contour, returnPoints=False)
    defects = cv2.convexityDefects(main_contour, hull_indices)
    defects_flat = defects.reshape(-1, 4)

    left_candidates, right_candidates = [], []
    for i in range(len(defects_flat)):
        _, _, far_idx, depth = defects_flat[i]
        depth, far_idx = int(depth), int(far_idx)
        if depth <= MIN_DEFECT_DEPTH:
            continue
        far_pt = main_contour[far_idx][0]
        x_pct = far_pt[0] / w_img * 100
        (left_candidates if x_pct < 50 else right_candidates).append((depth, far_idx))
    left_candidates.sort(reverse=True)
    right_candidates.sort(reverse=True)

    left_underarm = tuple(int(v) for v in main_contour[left_candidates[0][1]][0])
    right_underarm = tuple(int(v) for v in main_contour[right_candidates[0][1]][0])
    return left_underarm, right_underarm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Fetching golden garment and golden avatar...")
    garment_resp = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{GOLDEN_GARMENT_ID}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    garment_bytes = requests.get(garment_resp["clean_image_url"]).content

    avatar_resp = requests.get(
        f"http://127.0.0.1:8000/api/avatar/me?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    avatar_img_resp = requests.get(avatar_resp["processed_photo_url"])
    avatar_img = Image.open(io.BytesIO(avatar_img_resp.content)).convert("RGBA")
    aw, ah = avatar_img.size

    binary_mask, garment_img = get_binary_mask(garment_bytes)
    g_left_ua, g_right_ua = find_garment_underarms(binary_mask)
    g_hem_left, g_hem_right = get_garment_hem_corners(binary_mask, g_left_ua, g_right_ua)

    print(f"Garment: L_ua={g_left_ua} R_ua={g_right_ua} "
          f"hem_L={g_hem_left} hem_R={g_hem_right}")

    pose_keypoints = avatar_resp["pose_keypoints"]
    avatar_geometry = derive_avatar_geometry(pose_keypoints)

    a_left_ua = (avatar_geometry.left_underarm.x_pct / 100 * aw,
                  avatar_geometry.left_underarm.y_pct / 100 * ah)
    a_right_ua = (avatar_geometry.right_underarm.x_pct / 100 * aw,
                   avatar_geometry.right_underarm.y_pct / 100 * ah)

    a_left_hip = (pose_keypoints["left_hip"]["x"] * aw, pose_keypoints["left_hip"]["y"] * ah)
    a_right_hip = (pose_keypoints["right_hip"]["x"] * aw, pose_keypoints["right_hip"]["y"] * ah)

    a_hem_left = (a_left_ua[0], a_left_hip[1])
    a_hem_right = (a_right_ua[0], a_right_hip[1])

    print(f"Avatar: L_ua={a_left_ua} R_ua={a_right_ua} "
          f"hem_L={a_hem_left} hem_R={a_hem_right}")

    src_points = [g_left_ua, g_right_ua, g_hem_left, g_hem_right]

    # FIXED — order-agnostic offsetting, no assumption about which 
    # side has larger x. Use bounding box of all 4 destination 
    # points to compute a safe canvas + offset.
    raw_dst_points = [a_left_ua, a_right_ua, a_hem_left, a_hem_right]

    all_x = [p[0] for p in raw_dst_points]
    all_y = [p[1] for p in raw_dst_points]

    offset_x = -min(all_x) + 20
    offset_y = -min(all_y) + 20

    dst_points_local = [(p[0] + offset_x, p[1] + offset_y) for p in raw_dst_points]

    output_w = int(max(all_x) - min(all_x) + 40)
    output_h = int(max(all_y) - min(all_y) + 40)

    print(f"Output canvas: {output_w}x{output_h}  offset=({offset_x:.1f},{offset_y:.1f})")

    warped = piecewise_warp_garment(garment_img, src_points, dst_points_local, (output_w, output_h))

    result = avatar_img.copy()
    paste_x = int(min(all_x) - 20)
    paste_y = int(min(all_y) - 20)
    result.paste(warped, (paste_x, paste_y), warped)

    result.save(f"{OUTPUT_DIR}/tryon_result_piecewise.png")
    print(f"\nSaved: {OUTPUT_DIR}/tryon_result_piecewise.png")


if __name__ == "__main__":
    main()