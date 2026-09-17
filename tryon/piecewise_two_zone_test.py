"""
Two-region piecewise transform test. Splits the garment at its own
underarm row (real pixel split, not a bounding-box crop), applies
an independent similarity transform to each zone:

  Shoulder zone: garment shoulders -> avatar shoulders (2-point exact solve)
  Torso zone:    garment underarms -> avatar underarms (2-point exact solve,
                 same transform validated in correspondence_anchors_test.py)

Composites both zones together and reports the seam gap at the
boundary directly, without attempting to fix it.

Run: python -m tryon.piecewise_two_zone_test
"""

import io
import os
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw

from tryon.avatar_representation import build_avatar_representation
from tryon.avatar_torso_landmarks_test import find_semantic_underarm
from tryon.correspondence_shoulders_test import find_garment_underarms, find_garment_shoulder_points

GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
ALPHA_THRESHOLD = 10
OUTPUT_DIR = "outputs"


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD), image


def solve_similarity_2point(src_pts, dst_pts):
    """
    Exact similarity transform (scale+rotation+translation, 4 DOF)
    from exactly 2 point correspondences, via complex-number solve.
    Returns a 2x3 cv2-style affine matrix.
    """
    s0, s1 = complex(*src_pts[0]), complex(*src_pts[1])
    d0, d1 = complex(*dst_pts[0]), complex(*dst_pts[1])

    a = (d1 - d0) / (s1 - s0)
    b = d0 - a * s0

    matrix = np.array([
        [a.real, -a.imag, b.real],
        [a.imag,  a.real, b.imag]
    ])
    return matrix


def apply_matrix_to_point(matrix, pt):
    x, y = pt
    new_x = matrix[0,0]*x + matrix[0,1]*y + matrix[0,2]
    new_y = matrix[1,0]*x + matrix[1,1]*y + matrix[1,2]
    return (new_x, new_y)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Fetching golden garment...")
    garment_resp = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{GOLDEN_GARMENT_ID}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    garment_bytes = requests.get(garment_resp["clean_image_url"]).content
    garment_mask, garment_img = get_binary_mask(garment_bytes)
    gw, gh = garment_img.size
    garment_arr = np.array(garment_img)

    g_left_ua, g_right_ua = find_garment_underarms(garment_mask, gw)
    g_left_shoulder, g_right_shoulder = find_garment_shoulder_points(garment_mask, gw)
    print(f"Garment underarms: L={g_left_ua}  R={g_right_ua}")
    print(f"Garment shoulders: L={g_left_shoulder}  R={g_right_shoulder}")

    print("\nBuilding AvatarRepresentation...")
    rep = build_avatar_representation()
    aw, ah = rep.image.size

    a_left_underarm_y, _ = find_semantic_underarm(rep.torso_mask, rep.upper_arms_mask, aw, "left")
    a_right_underarm_y, _ = find_semantic_underarm(rep.torso_mask, rep.upper_arms_mask, aw, "right")
    a_left_row = rep.torso_mask[a_left_underarm_y]
    a_left_xs = np.where(a_left_row[:aw//2])[0]
    a_left_ua = (int(a_left_xs.min()), a_left_underarm_y)
    a_right_row = rep.torso_mask[a_right_underarm_y]
    a_right_xs = np.where(a_right_row[aw//2:])[0] + aw // 2
    a_right_ua = (int(a_right_xs.max()), a_right_underarm_y)

    print(f"Avatar semantic underarms: L={a_left_ua}  R={a_right_ua}")
    print(f"Avatar MediaPipe shoulders: L={rep.left_shoulder}  R={rep.right_shoulder}")

    # Pair by physical x-position
    g_small_ua = g_left_ua if g_left_ua[0] < g_right_ua[0] else g_right_ua
    g_large_ua = g_right_ua if g_left_ua[0] < g_right_ua[0] else g_left_ua
    g_small_sh = g_left_shoulder if g_left_shoulder[0] < g_right_shoulder[0] else g_right_shoulder
    g_large_sh = g_right_shoulder if g_left_shoulder[0] < g_right_shoulder[0] else g_left_shoulder

    a_small_ua = a_left_ua if a_left_ua[0] < a_right_ua[0] else a_right_ua
    a_large_ua = a_right_ua if a_left_ua[0] < a_right_ua[0] else a_left_ua
    a_small_sh = rep.left_shoulder if rep.left_shoulder[0] < rep.right_shoulder[0] else rep.right_shoulder
    a_large_sh = rep.right_shoulder if rep.left_shoulder[0] < rep.right_shoulder[0] else rep.left_shoulder

    # Two independent exact-solve transforms
    torso_matrix = solve_similarity_2point([g_small_ua, g_large_ua], [a_small_ua, a_large_ua])
    shoulder_matrix = solve_similarity_2point([g_small_sh, g_large_sh], [a_small_sh, a_large_sh])

    print(f"\nTorso transform matrix:\n{torso_matrix}")
    print(f"Shoulder transform matrix:\n{shoulder_matrix}")

    # Split garment at its own underarm row — real pixel split
    garment_underarm_y = (g_small_ua[1] + g_large_ua[1]) // 2
    print(f"\nSplitting garment at y={garment_underarm_y}")

    shoulder_zone_arr = garment_arr.copy()
    shoulder_zone_arr[garment_underarm_y:, :, 3] = 0  # zero alpha below split

    torso_zone_arr = garment_arr.copy()
    torso_zone_arr[:garment_underarm_y, :, 3] = 0  # zero alpha above split

    # Warp each zone independently into avatar-image-sized canvas
    warped_shoulder = cv2.warpAffine(
        shoulder_zone_arr, shoulder_matrix, (aw, ah),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT
    )
    warped_torso = cv2.warpAffine(
        torso_zone_arr, torso_matrix, (aw, ah),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT
    )

    # Composite: torso first, then shoulder on top (shoulder zone 
    # naturally sits above in y, minimal overlap expected)
    result_arr = np.array(rep.image.convert("RGBA")).astype(np.uint8)
    for layer in [warped_torso, warped_shoulder]:
        alpha = layer[:, :, 3:4].astype(np.float32) / 255.0
        result_arr = (result_arr.astype(np.float32) * (1 - alpha) + layer.astype(np.float32) * alpha).astype(np.uint8)

    result = Image.fromarray(result_arr)

    # Check seam: where does the transformed underarm anchor from the 
    # TORSO transform land, vs where the SHOULDER transform's own 
    # bottom edge lands, at the split line
    torso_underarm_check = apply_matrix_to_point(torso_matrix, g_small_ua)
    shoulder_bottom_check = apply_matrix_to_point(shoulder_matrix, (g_small_ua[0], garment_underarm_y))
    seam_gap = np.hypot(torso_underarm_check[0]-shoulder_bottom_check[0],
                          torso_underarm_check[1]-shoulder_bottom_check[1])
    print(f"\nSeam gap at split line (small-x side): {seam_gap:.1f}px")

    draw = ImageDraw.Draw(result)
    for pt, color in [(a_small_ua, (255,0,255)), (a_large_ua, (255,0,255)),
                        (a_small_sh, (0,255,0)), (a_large_sh, (0,255,0))]:
        x, y = pt
        draw.ellipse([x-5,y-5,x+5,y+5], outline=color, width=2)

    result.convert("RGB").save(f"{OUTPUT_DIR}/piecewise_two_zone_test.png")
    print(f"\nSaved: {OUTPUT_DIR}/piecewise_two_zone_test.png")
    print(f"\n>>> DECISION GATE: does the composited result show good "
          f"alignment at BOTH underarms and shoulders, with an acceptable "
          f"seam at the boundary — or is there a visible gap/discontinuity "
          f"at the split line requiring a denser mesh instead?")


if __name__ == "__main__":
    main()