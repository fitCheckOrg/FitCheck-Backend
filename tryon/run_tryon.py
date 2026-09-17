"""
End-to-end vertical slice. ONE golden garment, ONE golden avatar.
Produces avatar_original.png, garment_original.png, tryon_result.png
for direct visual inspection.

Fixed this pass:
1. garment_warper.py's crop now restricts to actual underarm span
2. Avatar anchor now uses derive_avatar_geometry() (the validated 
   underarm-interpolation module) instead of raw shoulder position

Run: python -m tryon.run_tryon
"""

import io
import os
import requests
import numpy as np
import cv2
from PIL import Image

from tryon.garment_mapper import get_garment_coordinate_system
from tryon.garment_warper import warp_garment_simple
from tryon.compositor import composite
from workers.avatar.geometry_derivation import derive_avatar_geometry

ALPHA_THRESHOLD = 10
MIN_DEFECT_DEPTH = 1000

GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"  # polo_baseline
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"

OUTPUT_DIR = "outputs"


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255, image


def find_garment_underarms_and_hem(binary_mask):
    """Reusing exactly the validated Layer 0 + hem logic."""
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
    hem_point = tuple(int(v) for v in main_contour[lowest_idx][0])

    return left_underarm, right_underarm, hem_point


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

    # --- Garment geometry (unchanged, already validated) ---
    binary_mask, garment_img = get_binary_mask(garment_bytes)
    g_left_ua, g_right_ua, g_hem = find_garment_underarms_and_hem(binary_mask)

    garment_coords = get_garment_coordinate_system(g_left_ua, g_right_ua, g_hem)
    print(f"Garment: underarms={g_left_ua},{g_right_ua}  hem={g_hem}  "
          f"width={garment_coords['width']:.1f}  height={garment_coords['height']:.1f}")

    # --- Avatar geometry — FIXED: use derive_avatar_geometry(), 
    # not raw shoulder position ---
    pose_keypoints = avatar_resp["pose_keypoints"]
    avatar_geometry = derive_avatar_geometry(pose_keypoints)

    a_left_underarm = (avatar_geometry.left_underarm.x_pct / 100 * aw,
                         avatar_geometry.left_underarm.y_pct / 100 * ah)
    a_right_underarm = (avatar_geometry.right_underarm.x_pct / 100 * aw,
                          avatar_geometry.right_underarm.y_pct / 100 * ah)

    avatar_anchor = ((a_left_underarm[0] + a_right_underarm[0]) / 2,
                       (a_left_underarm[1] + a_right_underarm[1]) / 2)
    avatar_width = np.hypot(a_right_underarm[0] - a_left_underarm[0],
                              a_right_underarm[1] - a_left_underarm[1])

    # Height reference: underarm to hip, using the same pose_keypoints
    a_left_hip = (pose_keypoints["left_hip"]["x"] * aw, pose_keypoints["left_hip"]["y"] * ah)
    a_right_hip = (pose_keypoints["right_hip"]["x"] * aw, pose_keypoints["right_hip"]["y"] * ah)
    avatar_hip_mid_y = (a_left_hip[1] + a_right_hip[1]) / 2
    avatar_height = avatar_hip_mid_y - avatar_anchor[1]

    avatar_coords = {"anchor": avatar_anchor, "width": avatar_width, "height": avatar_height}
    print(f"Avatar: underarm_mid={avatar_anchor}  width={avatar_width:.1f}  height={avatar_height:.1f}")

    # --- Warp (FIXED crop) + composite ---
    warped = warp_garment_simple(garment_img, garment_coords, avatar_coords)
    result = composite(avatar_img, warped, avatar_coords["anchor"])

    garment_img.save(f"{OUTPUT_DIR}/garment_original.png")
    avatar_img.save(f"{OUTPUT_DIR}/avatar_original.png")
    result.save(f"{OUTPUT_DIR}/tryon_result.png")

    print(f"\nSaved: {OUTPUT_DIR}/garment_original.png")
    print(f"Saved: {OUTPUT_DIR}/avatar_original.png")
    print(f"Saved: {OUTPUT_DIR}/tryon_result.png")
    print(f"\n>>> Inspect tryon_result.png — does the garment follow the torso, "
          f"or does it look like a rectangular sticker?")


if __name__ == "__main__":
    main()