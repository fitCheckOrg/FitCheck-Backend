"""
Phase C — First torso fitting experiment.

Tests Hypothesis H1: after underarm anchoring establishes torso 
position/width, garment vertical extent is determined by the 
garment's OWN intrinsic height/width proportion — not mapped to 
any avatar landmark (no invented "avatar hem point").

Uses ONLY validated pieces from tonight:
  - Contour + underarm/hem detection (Phase A geometry, already proven)
  - AvatarGeometryDerivation (already built, already tested)

Rotation deliberately NOT applied — translation + uniform scale 
only, per the explicit decision to keep it measurable but disabled 
until evidence shows it's needed. Underarm-pair MIDPOINT anchors 
translation; underarm-pair DISTANCE sets scale. Individual point 
alignment is therefore approximate, not exact — this is honest, 
not a bug, and is exactly what this experiment checks.

Run: python fitting_experiment_h1.py
"""

import io
import math
import requests
import numpy as np
import cv2
from PIL import Image

GARMENT_ITEM_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"  # polo_baseline — 
                                                              # visually confirmed 
                                                              # clean underarm + 
                                                              # hem all night
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"

ALPHA_THRESHOLD = 10
MIN_DEFECT_DEPTH = 1000
UNDERARM_T = 0.13  # AvatarGeometryDerivation's working hypothesis


# ---- Garment-side: reuse validated Phase A contour geometry ----

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


# ---- Avatar-side: reuse the already-built, already-tested module ----

def derive_avatar_underarms(pose_keypoints):
    def interpolate(shoulder, hip, t):
        x = shoulder["x"] + t * (hip["x"] - shoulder["x"])
        y = shoulder["y"] + t * (hip["y"] - shoulder["y"])
        return x, y

    left = interpolate(pose_keypoints["left_shoulder"], pose_keypoints["left_hip"], UNDERARM_T)
    right = interpolate(pose_keypoints["right_shoulder"], pose_keypoints["right_hip"], UNDERARM_T)
    return left, right  # normalized 0-1, converted to pixels by caller


def main():
    print("Fetching real garment and avatar...")

    garment_resp = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{GARMENT_ITEM_ID}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    garment_bytes = requests.get(garment_resp["clean_image_url"]).content

    avatar_resp = requests.get(
        f"http://127.0.0.1:8000/api/avatar/me?user_id={AVATAR_USER_ID}"
    ).json()["data"]

    # --- DIAGNOSTIC ---
    print("\n=== RAW AVATAR RESPONSE DEBUG ===")
    print(f"avatar_id: {avatar_resp.get('avatar_id')}")
    print(f"avatar_version: {avatar_resp.get('avatar_version')}")
    print(f"processed_photo_url: {avatar_resp.get('processed_photo_url')}")
    print(f"left_shoulder: {avatar_resp['pose_keypoints'].get('left_shoulder')}")
    print(f"right_shoulder: {avatar_resp['pose_keypoints'].get('right_shoulder')}")
    print(f"left_hip: {avatar_resp['pose_keypoints'].get('left_hip')}")
    print(f"right_hip: {avatar_resp['pose_keypoints'].get('right_hip')}")
    print("=== END DEBUG ===\n")

    avatar_img_resp = requests.get(avatar_resp["processed_photo_url"])
    avatar_img = Image.open(io.BytesIO(avatar_img_resp.content)).convert("RGBA")
    aw, ah = avatar_img.size
    print(f"Actual downloaded avatar image dimensions: {aw} x {ah}")

    # --- Garment geometry ---
    binary_mask, garment_img = get_binary_mask(garment_bytes)
    g_left_ua, g_right_ua, g_hem = find_garment_underarms_and_hem(binary_mask)

    g_underarm_width = math.hypot(g_right_ua[0] - g_left_ua[0], g_right_ua[1] - g_left_ua[1])
    g_underarm_mid = ((g_left_ua[0] + g_right_ua[0]) / 2, (g_left_ua[1] + g_right_ua[1]) / 2)
    g_intrinsic_height = g_hem[1] - g_underarm_mid[1]
    aspect_ratio = g_intrinsic_height / g_underarm_width

    print(f"\n--- Garment intrinsic geometry ---")
    print(f"underarm_width: {g_underarm_width:.1f}px")
    print(f"underarm_to_hem_height: {g_intrinsic_height:.1f}px")
    print(f"aspect_ratio (height/width): {aspect_ratio:.3f}")

    # --- Avatar geometry ---
    a_left_ua_norm, a_right_ua_norm = derive_avatar_underarms(avatar_resp["pose_keypoints"])
    a_left_ua = (a_left_ua_norm[0] * aw, a_left_ua_norm[1] * ah)
    a_right_ua = (a_right_ua_norm[0] * aw, a_right_ua_norm[1] * ah)

    a_underarm_width = math.hypot(a_right_ua[0] - a_left_ua[0], a_right_ua[1] - a_left_ua[1])
    a_underarm_mid = ((a_left_ua[0] + a_right_ua[0]) / 2, (a_left_ua[1] + a_right_ua[1]) / 2)

    print(f"\n--- Avatar target geometry ---")
    print(f"underarm_width: {a_underarm_width:.1f}px")
    print(f"underarm_midpoint: ({a_underarm_mid[0]:.1f}, {a_underarm_mid[1]:.1f})")

    # --- H1: fitted height from garment's OWN proportion ---
    scale = a_underarm_width / g_underarm_width
    fitted_height = aspect_ratio * a_underarm_width

    print(f"\n--- Fitting result (H1) ---")
    print(f"scale (uniform): {scale:.3f}")
    print(f"fitted_width: {a_underarm_width:.1f}px")
    print(f"fitted_height: {fitted_height:.1f}px")

    crop_top = int(g_underarm_mid[1])
    crop_bottom = int(g_hem[1])
    torso_crop = garment_img.crop((0, crop_top, garment_img.width, crop_bottom))

    resized = torso_crop.resize((int(a_underarm_width * 1.15), int(fitted_height * 1.15)), Image.LANCZOS)

    result = avatar_img.copy()
    paste_x = int(a_underarm_mid[0] - resized.width / 2)
    paste_y = int(a_underarm_mid[1])
    result.paste(resized, (paste_x, paste_y), resized)
    result.save("fitting_experiment_h1.png")
    print(f"\nSaved: fitting_experiment_h1.png")


if __name__ == "__main__":
    main()