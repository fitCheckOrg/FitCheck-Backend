"""
H1.1 — Real Torso Silhouette experiment.

Changes EXACTLY ONE variable from H1: replaces the rectangular
torso crop with the garment's actual alpha-mask silhouette, 
restricted to the torso's horizontal column range (excluding 
sleeves by masking, not by losing the real curved edges within 
that range).

Same avatar, same garment, same GarmentLandmarkSet-derived 
underarm/hem points, same H1 placement math, same scale. The 
ONLY thing different: what gets pasted is a real silhouette 
with transparent background intact, not an opaque rectangle.

Run: python fitting_experiment_h1_1.py
"""

import io
import math
import requests
import numpy as np
import cv2
from PIL import Image

GARMENT_ITEM_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"

ALPHA_THRESHOLD = 10
MIN_DEFECT_DEPTH = 1000
UNDERARM_T = 0.13


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255, image


def find_garment_underarms_and_hem(binary_mask):
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

    lowest_idx = np.argmax(main_contour[:, 0, 1])
    hem_bottom = tuple(int(v) for v in main_contour[lowest_idx][0])

    return left_underarm, right_underarm, hem_bottom


def extract_torso_silhouette(garment_img, left_underarm, right_underarm, hem_y):
    """
    THE ONLY NEW OPERATION vs H1. Instead of a rectangular crop,
    zero out alpha OUTSIDE the torso's horizontal column range
    (excluding sleeves by column, matching H1's original torso
    boundary definition) and outside the vertical underarm-to-hem
    range — but WITHIN that box, preserve the garment's real alpha
    values exactly as they are. Real curved edges stay real; only
    the sleeve columns and out-of-range rows get masked away.
    """
    rgba = np.array(garment_img).copy()
    h, w = rgba.shape[:2]

    left_x = min(left_underarm[0], right_underarm[0])
    right_x = max(left_underarm[0], right_underarm[0])
    top_y = min(left_underarm[1], right_underarm[1])

    mask = np.zeros((h, w), dtype=bool)
    mask[top_y:hem_y, left_x:right_x] = True

    rgba[:, :, 3] = np.where(mask, rgba[:, :, 3], 0)

    return Image.fromarray(rgba, mode="RGBA")


def derive_avatar_underarms(pose_keypoints):
    def interpolate(shoulder, hip, t):
        x = shoulder["x"] + t * (hip["x"] - shoulder["x"])
        y = shoulder["y"] + t * (hip["y"] - shoulder["y"])
        return x, y
    left = interpolate(pose_keypoints["left_shoulder"], pose_keypoints["left_hip"], UNDERARM_T)
    right = interpolate(pose_keypoints["right_shoulder"], pose_keypoints["right_hip"], UNDERARM_T)
    return left, right


def main():
    print("Fetching real garment and avatar...")

    garment_resp = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{GARMENT_ITEM_ID}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    garment_bytes = requests.get(garment_resp["clean_image_url"]).content

    avatar_resp = requests.get(
        f"http://127.0.0.1:8000/api/avatar/me?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    avatar_img_resp = requests.get(avatar_resp["processed_photo_url"])
    avatar_img = Image.open(io.BytesIO(avatar_img_resp.content)).convert("RGBA")
    aw, ah = avatar_img.size
    print(f"Avatar image dimensions: {aw} x {ah}")

    binary_mask, garment_img = get_binary_mask(garment_bytes)
    g_left_ua, g_right_ua, g_hem = find_garment_underarms_and_hem(binary_mask)

    g_underarm_width = math.hypot(g_right_ua[0] - g_left_ua[0], g_right_ua[1] - g_left_ua[1])
    g_underarm_mid = ((g_left_ua[0] + g_right_ua[0]) / 2, (g_left_ua[1] + g_right_ua[1]) / 2)
    g_intrinsic_height = g_hem[1] - g_underarm_mid[1]
    aspect_ratio = g_intrinsic_height / g_underarm_width

    print(f"\n--- Garment intrinsic geometry (same as H1) ---")
    print(f"underarm_width: {g_underarm_width:.1f}px")
    print(f"underarm_to_hem_height: {g_intrinsic_height:.1f}px")
    print(f"aspect_ratio: {aspect_ratio:.3f}")

    # NEW: extract real torso silhouette instead of rectangular crop
    torso_silhouette = extract_torso_silhouette(garment_img, g_left_ua, g_right_ua, g_hem[1])
    mask_arr = np.array(torso_silhouette)[:, :, 3]
    total_box_pixels = (g_hem[1] - min(g_left_ua[1], g_right_ua[1])) * (max(g_left_ua[0], g_right_ua[0]) - min(g_left_ua[0], g_right_ua[0]))
    opaque_pixels = np.sum(mask_arr > ALPHA_THRESHOLD)
    print(f"\nSilhouette check: {opaque_pixels} opaque px out of {total_box_pixels} box px "
      f"({opaque_pixels/total_box_pixels*100:.1f}% filled)")
    torso_crop = torso_silhouette.crop((0, g_underarm_mid[1].__int__(), garment_img.width, g_hem[1]))

    a_left_ua_norm, a_right_ua_norm = derive_avatar_underarms(avatar_resp["pose_keypoints"])
    a_left_ua = (a_left_ua_norm[0] * aw, a_left_ua_norm[1] * ah)
    a_right_ua = (a_right_ua_norm[0] * aw, a_right_ua_norm[1] * ah)

    a_underarm_width = math.hypot(a_right_ua[0] - a_left_ua[0], a_right_ua[1] - a_left_ua[1])
    a_underarm_mid = ((a_left_ua[0] + a_right_ua[0]) / 2, (a_left_ua[1] + a_right_ua[1]) / 2)

    print(f"\n--- Avatar target geometry (same as H1) ---")
    print(f"underarm_width: {a_underarm_width:.1f}px")
    print(f"underarm_midpoint: ({a_underarm_mid[0]:.1f}, {a_underarm_mid[1]:.1f})")

    fitted_height = aspect_ratio * a_underarm_width
    print(f"\n--- Fitting result (identical math to H1) ---")
    print(f"fitted_width: {a_underarm_width:.1f}px")
    print(f"fitted_height: {fitted_height:.1f}px")

    resized = torso_crop.resize((int(a_underarm_width * 1.15), int(fitted_height * 1.15)), Image.LANCZOS)

    result = avatar_img.copy()
    paste_x = int(a_underarm_mid[0] - resized.width / 2)
    paste_y = int(a_underarm_mid[1])
    result.paste(resized, (paste_x, paste_y), resized)
    result.save("fitting_experiment_h1_1.png")
    print(f"\nSaved: fitting_experiment_h1_1.png")


if __name__ == "__main__":
    main()