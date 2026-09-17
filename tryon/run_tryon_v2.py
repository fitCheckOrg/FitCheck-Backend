"""
V2 — contour-driven piecewise affine warp. Both garment AND avatar
side boundaries sampled from REAL contours (avatar's processed
photo has a real alpha channel from its own remove.bg pass — no
fabricated straight-line envelope).

Run: python -m tryon.run_tryon_v2
"""

import io
import os
import requests
import numpy as np
import cv2
from PIL import Image

from tryon.contour_sampler import sample_path_between
from tryon.piecewise_warper import piecewise_warp_garment_strip
from workers.avatar.geometry_derivation import derive_avatar_geometry

ALPHA_THRESHOLD = 10
MIN_DEFECT_DEPTH = 1000
NUM_SAMPLES = 5  # underarm + hem + 3 intermediate points per side

GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
OUTPUT_DIR = "outputs"


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255, image


def get_main_contour(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea)


def find_garment_underarms(main_contour, w_img):
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


def get_hem_corners(main_contour):
    hem_y_target = int(main_contour[:, 0, 1].max())
    near_hem = main_contour[np.abs(main_contour[:, 0, 1] - hem_y_target) < 15]
    xs = near_hem[:, 0, 0]
    return (int(xs.min()), hem_y_target), (int(xs.max()), hem_y_target)


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
    avatar_bytes = avatar_img_resp.content
    aw_img = Image.open(io.BytesIO(avatar_bytes)).convert("RGBA")
    aw, ah = aw_img.size

    # --- Garment: real contour, real underarms/hem ---
    g_mask, garment_img = get_binary_mask(garment_bytes)
    g_contour = get_main_contour(g_mask)
    g_left_ua, g_right_ua = find_garment_underarms(g_contour, g_mask.shape[1])
    g_hem_left, g_hem_right = get_hem_corners(g_contour)

    g_left_chain = sample_path_between(g_contour, g_left_ua, g_hem_left, NUM_SAMPLES)
    g_right_chain = sample_path_between(g_contour, g_right_ua, g_hem_right, NUM_SAMPLES)

    print(f"Garment left chain: {g_left_chain}")
    print(f"Garment right chain: {g_right_chain}")

    # --- Avatar: REAL contour from its own alpha channel ---
    a_mask, _ = get_binary_mask(avatar_bytes)
    a_contour = get_main_contour(a_mask)

    pose_keypoints = avatar_resp["pose_keypoints"]
    avatar_geometry = derive_avatar_geometry(pose_keypoints)

    a_left_ua = (avatar_geometry.left_underarm.x_pct / 100 * aw,
                  avatar_geometry.left_underarm.y_pct / 100 * ah)
    a_right_ua = (avatar_geometry.right_underarm.x_pct / 100 * aw,
                   avatar_geometry.right_underarm.y_pct / 100 * ah)
    a_left_hip = (pose_keypoints["left_hip"]["x"] * aw, pose_keypoints["left_hip"]["y"] * ah)
    a_right_hip = (pose_keypoints["right_hip"]["x"] * aw, pose_keypoints["right_hip"]["y"] * ah)

    a_left_chain = sample_path_between(a_contour, a_left_ua, a_left_hip, NUM_SAMPLES)
    a_right_chain = sample_path_between(a_contour, a_right_ua, a_right_hip, NUM_SAMPLES)

    print(f"Avatar left chain: {a_left_chain}")
    print(f"Avatar right chain: {a_right_chain}")

    # --- Canvas + offset, based on actual avatar chain bounding box ---
    all_pts = a_left_chain + a_right_chain
    all_x = [p[0] for p in all_pts]
    all_y = [p[1] for p in all_pts]

    offset_x = -min(all_x) + 20
    offset_y = -min(all_y) + 20

    a_left_chain_local = [(p[0] + offset_x, p[1] + offset_y) for p in a_left_chain]
    a_right_chain_local = [(p[0] + offset_x, p[1] + offset_y) for p in a_right_chain]

    output_w = int(max(all_x) - min(all_x) + 40)
    output_h = int(max(all_y) - min(all_y) + 40)

    warped = piecewise_warp_garment_strip(
        garment_img, g_left_chain, g_right_chain,
        a_left_chain_local, a_right_chain_local, (output_w, output_h)
    )

    result = aw_img.copy()
    paste_x = int(min(all_x) - 20)
    paste_y = int(min(all_y) - 20)
    result.paste(warped, (paste_x, paste_y), warped)

    result.save(f"{OUTPUT_DIR}/tryon_result_v2.png")
    print(f"\nSaved: {OUTPUT_DIR}/tryon_result_v2.png")


if __name__ == "__main__":
    main()