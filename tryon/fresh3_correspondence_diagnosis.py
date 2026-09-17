"""
Diagnostic-only trace for fresh_3_t-shirt's placement failure.
Prints every intermediate value from garment landmarks through to
final render coordinates, unchanged from the actual pipeline logic
used in garment_generalization_test.py. No thresholds changed, no
new heuristics, no fitting logic modified.

Run: python -m tryon.fresh3_correspondence_diagnosis
"""

import io
import os
import requests
import numpy as np
import cv2
from PIL import Image

from tryon.avatar_representation import build_avatar_representation
from tryon.avatar_torso_landmarks_test import find_semantic_underarm
from tryon.correspondence_shoulders_test import find_garment_underarms
from tryon.garment_shoulder_region_extraction import get_binary_mask
from tryon.piecewise_continuous_test import get_garment_shoulder_points

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
FRESH_3_ID = "5c22b2ea-fedc-4dbb-b649-fc645774d011"  # fresh_3_t-shirt
OUTPUT_DIR = "outputs"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== STAGE 1: GARMENT LANDMARKS ===\n")
    garment_resp = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{FRESH_3_ID}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    garment_bytes = requests.get(garment_resp["clean_image_url"]).content
    garment_mask, garment_img = get_binary_mask(garment_bytes)
    gw, gh = garment_img.size
    print(f"Garment image dimensions: {gw} x {gh}")

    g_left_ua, g_right_ua = find_garment_underarms(garment_mask, gw)
    print(f"Garment underarms: L={g_left_ua}  R={g_right_ua}")

    g_left_sh, g_right_sh = get_garment_shoulder_points(garment_mask, garment_img, g_left_ua, g_right_ua, gw)
    print(f"Garment shoulders: L={g_left_sh}  R={g_right_sh}")

    # Sanity check: do these landmarks make sense relative to the 
    # garment's own dimensions? (pure inspection, no correction)
    print(f"\nSanity ratios (garment-space):")
    print(f"  Underarm width: {abs(g_right_ua[0]-g_left_ua[0])}px  "
          f"({abs(g_right_ua[0]-g_left_ua[0])/gw*100:.1f}% of image width)")
    print(f"  Shoulder width: {abs(g_right_sh[0]-g_left_sh[0])}px  "
          f"({abs(g_right_sh[0]-g_left_sh[0])/gw*100:.1f}% of image width)")
    print(f"  Underarm Y: L={g_left_ua[1]} R={g_right_ua[1]}  "
          f"({g_left_ua[1]/gh*100:.1f}% / {g_right_ua[1]/gh*100:.1f}% down image height)")
    print(f"  Shoulder Y: L={g_left_sh[1]} R={g_right_sh[1]}  "
          f"({g_left_sh[1]/gh*100:.1f}% / {g_right_sh[1]/gh*100:.1f}% down image height)")

    print(f"\n=== STAGE 2: AVATAR LANDMARKS ===\n")
    rep = build_avatar_representation()
    aw, ah = rep.image.size
    print(f"Avatar image dimensions: {aw} x {ah}")

    a_l_uy, _ = find_semantic_underarm(rep.torso_mask, rep.upper_arms_mask, aw, "left")
    a_r_uy, _ = find_semantic_underarm(rep.torso_mask, rep.upper_arms_mask, aw, "right")
    a_l_row = rep.torso_mask[a_l_uy]; a_l_xs = np.where(a_l_row[:aw//2])[0]
    a_left_ua = (int(a_l_xs.min()), a_l_uy)
    a_r_row = rep.torso_mask[a_r_uy]; a_r_xs = np.where(a_r_row[aw//2:])[0] + aw//2
    a_right_ua = (int(a_r_xs.max()), a_r_uy)

    print(f"Avatar underarms: L={a_left_ua}  R={a_right_ua}")
    print(f"Avatar shoulders: L={rep.left_shoulder}  R={rep.right_shoulder}")

    print(f"\n=== STAGE 3: CORRESPONDENCE (pairing by physical x-position) ===\n")
    g_s_ua = g_left_ua if g_left_ua[0]<g_right_ua[0] else g_right_ua
    g_l_ua = g_right_ua if g_left_ua[0]<g_right_ua[0] else g_left_ua
    g_s_sh = g_left_sh if g_left_sh[0]<g_right_sh[0] else g_right_sh
    g_l_sh = g_right_sh if g_left_sh[0]<g_right_sh[0] else g_left_sh
    a_s_ua = a_left_ua if a_left_ua[0]<a_right_ua[0] else a_right_ua
    a_l_ua = a_right_ua if a_left_ua[0]<a_right_ua[0] else a_left_ua
    a_s_sh = rep.left_shoulder if rep.left_shoulder[0]<rep.right_shoulder[0] else rep.right_shoulder
    a_l_sh = rep.right_shoulder if rep.left_shoulder[0]<rep.right_shoulder[0] else rep.left_shoulder

    print(f"PAIRING:")
    print(f"  garment_small_x_underarm {g_s_ua}  →  avatar_small_x_underarm {a_s_ua}")
    print(f"  garment_large_x_underarm {g_l_ua}  →  avatar_large_x_underarm {a_l_ua}")
    print(f"  garment_small_x_shoulder {g_s_sh}  →  avatar_small_x_shoulder {a_s_sh}")
    print(f"  garment_large_x_shoulder {g_l_sh}  →  avatar_large_x_shoulder {a_l_sh}")

    src_tris = [[g_s_sh, g_l_sh, g_s_ua], [g_l_sh, g_s_ua, g_l_ua]]
    dst_tris = [[a_s_sh, a_l_sh, a_s_ua], [a_l_sh, a_s_ua, a_l_ua]]

    print(f"\nTRIANGLE 1 (upper): src={src_tris[0]}  dst={dst_tris[0]}")
    print(f"TRIANGLE 2 (lower): src={src_tris[1]}  dst={dst_tris[1]}")

    # Compute the actual affine matrix cv2 would use for each triangle
    for i, (src_tri, dst_tri) in enumerate(zip(src_tris, dst_tris)):
        src = np.array(src_tri, dtype=np.float32)
        dst = np.array(dst_tri, dtype=np.float32)
        matrix = cv2.getAffineTransform(src, dst)
        print(f"\nTriangle {i+1} affine matrix:\n{matrix}")
        scale_x = np.hypot(matrix[0,0], matrix[1,0])
        scale_y = np.hypot(matrix[0,1], matrix[1,1])
        print(f"  Implied scale: x={scale_x:.4f}  y={scale_y:.4f}")

    print(f"\n=== STAGE 4: WHERE DO GARMENT CORNERS ACTUALLY LAND? ===\n")
    # Map the garment's own top-left and top-right corners through 
    # each triangle's transform, to see where the WHOLE garment 
    # bounding box ends up, not just the anchor points
    for label, pt in [("garment (0,0) top-left", (0,0)), ("garment (w,0) top-right", (gw,0))]:
        src = np.array(src_tris[0], dtype=np.float32)
        dst = np.array(dst_tris[0], dtype=np.float32)
        matrix = cv2.getAffineTransform(src, dst)
        x, y = pt
        mapped_x = matrix[0,0]*x + matrix[0,1]*y + matrix[0,2]
        mapped_y = matrix[1,0]*x + matrix[1,1]*y + matrix[1,2]
        print(f"  {label} {pt}  →  ({mapped_x:.1f}, {mapped_y:.1f})  "
              f"[avatar image is {aw}x{ah}]")

    print(f"\n>>> DIAGNOSIS: compare garment landmark Y-positions (Stage 1) "
          f"against garment image height — if underarms/shoulders sit at an "
          f"unusual fraction of the image (e.g. very near the top), the "
          f"garment photo itself may have unusual proportions/cropping "
          f"that the pipeline doesn't account for.")


if __name__ == "__main__":
    main()