"""
Visualizes the garment↔avatar correspondence mesh directly — both
point sets drawn on their own images, connected in matching order,
so we can inspect whether the correspondence is anatomically
sensible BEFORE increasing sample density or touching hem/hip.

No warping. No compositing. Pure correspondence inspection.

Run: python -m tryon.correspondence_mesh_diagnostic
"""

import io
import os
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw

from tryon.avatar_torso_sampler import sample_torso_side
from tryon.contour_sampler import sample_path_between
from workers.avatar.geometry_derivation import derive_avatar_geometry
from tryon.avatar_torso_sampler_v2 import sample_torso_side_arm_aware

ALPHA_THRESHOLD = 10
MIN_DEFECT_DEPTH = 1000
NUM_SAMPLES = 5

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


def draw_mesh(image, left_chain, right_chain, title_color=(255, 255, 0)):
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    for i, (l, r) in enumerate(zip(left_chain, right_chain)):
        lx, ly = int(l[0]), int(l[1])
        rx, ry = int(r[0]), int(r[1])
        draw.ellipse([lx-6, ly-6, lx+6, ly+6], outline=(255, 0, 0), width=3)
        draw.text((lx+8, ly-6), f"L{i}", fill=(255, 0, 0))
        draw.ellipse([rx-6, ry-6, rx+6, ry+6], outline=(0, 150, 255), width=3)
        draw.text((rx+8, ry-6), f"R{i}", fill=(0, 150, 255))
        draw.line([lx, ly, rx, ry], fill=title_color, width=1)

        if i < len(left_chain) - 1:
            l2x, l2y = int(left_chain[i+1][0]), int(left_chain[i+1][1])
            r2x, r2y = int(right_chain[i+1][0]), int(right_chain[i+1][1])
            draw.line([lx, ly, l2x, l2y], fill=(255, 100, 100), width=1)
            draw.line([rx, ry, r2x, r2y], fill=(100, 200, 255), width=1)

    return canvas


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    garment_resp = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{GOLDEN_GARMENT_ID}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    garment_bytes = requests.get(garment_resp["clean_image_url"]).content

    avatar_resp = requests.get(
        f"http://127.0.0.1:8000/api/avatar/me?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    avatar_bytes = requests.get(avatar_resp["processed_photo_url"]).content
    avatar_img = Image.open(io.BytesIO(avatar_bytes)).convert("RGBA")
    aw, ah = avatar_img.size

    g_mask, garment_img = get_binary_mask(garment_bytes)
    g_contour = get_main_contour(g_mask)
    g_left_ua, g_right_ua = find_garment_underarms(g_contour, g_mask.shape[1])
    g_hem_left, g_hem_right = get_hem_corners(g_contour)
    g_left_chain = sample_path_between(g_contour, g_left_ua, g_hem_left, NUM_SAMPLES)
    g_right_chain = sample_path_between(g_contour, g_right_ua, g_hem_right, NUM_SAMPLES)

    a_mask, _ = get_binary_mask(avatar_bytes)
    a_contour = get_main_contour(a_mask)
    pose_keypoints = avatar_resp["pose_keypoints"]
    avatar_geometry = derive_avatar_geometry(pose_keypoints)
    a_left_ua = (avatar_geometry.left_underarm.x_pct / 100 * aw, avatar_geometry.left_underarm.y_pct / 100 * ah)
    a_right_ua = (avatar_geometry.right_underarm.x_pct / 100 * aw, avatar_geometry.right_underarm.y_pct / 100 * ah)
    a_left_hip = (pose_keypoints["left_hip"]["x"] * aw, pose_keypoints["left_hip"]["y"] * ah)
    a_right_hip = (pose_keypoints["right_hip"]["x"] * aw, pose_keypoints["right_hip"]["y"] * ah)
    a_left_shoulder = (pose_keypoints["left_shoulder"]["x"] * aw, pose_keypoints["left_shoulder"]["y"] * ah)
    a_right_shoulder = (pose_keypoints["right_shoulder"]["x"] * aw, pose_keypoints["right_shoulder"]["y"] * ah)
    a_left_elbow = (pose_keypoints["left_elbow"]["x"] * aw, pose_keypoints["left_elbow"]["y"] * ah)
    a_right_elbow = (pose_keypoints["right_elbow"]["x"] * aw, pose_keypoints["right_elbow"]["y"] * ah)
    a_left_wrist = (pose_keypoints["left_wrist"]["x"] * aw, pose_keypoints["left_wrist"]["y"] * ah)
    a_right_wrist = (pose_keypoints["right_wrist"]["x"] * aw, pose_keypoints["right_wrist"]["y"] * ah)

    a_left_chain = sample_torso_side_arm_aware(
        a_contour, a_left_ua, a_left_hip, a_left_shoulder, a_left_elbow, a_left_wrist,
        is_left_side=True, num_samples=NUM_SAMPLES
    )
    a_right_chain = sample_torso_side_arm_aware(
        a_contour, a_right_ua, a_right_hip, a_right_shoulder, a_right_elbow, a_right_wrist,
        is_left_side=False, num_samples=NUM_SAMPLES
    )
    print("GARMENT chains:")
    print(f"  Left:  {g_left_chain}")
    print(f"  Right: {g_right_chain}")
    print("\nAVATAR chains (paired: garment_right <-> avatar_left, garment_left <-> avatar_right, per earlier swap):")
    print(f"  Left:  {a_left_chain}")
    print(f"  Right: {a_right_chain}")

    garment_mesh = draw_mesh(garment_img, g_left_chain, g_right_chain)
    # Using the SAME swapped pairing as the actual warp, so the 
    # visualized correspondence matches what's really being warped
    avatar_mesh = draw_mesh(avatar_img, a_right_chain, a_left_chain)

    garment_mesh.save(f"{OUTPUT_DIR}/mesh_garment.png")
    avatar_mesh.save(f"{OUTPUT_DIR}/mesh_avatar.png")
    print(f"\nSaved: {OUTPUT_DIR}/mesh_garment.png")
    print(f"Saved: {OUTPUT_DIR}/mesh_avatar.png")


if __name__ == "__main__":
    main()