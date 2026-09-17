"""
Re-tests global affine and two-zone independent piecewise using the
CORRECTED, region-based shoulder junction — not the rejected collar
proxy. Previous 0.407 vs 0.833 scale-ratio finding is treated as
obsolete, not reused.

Run: python -m tryon.correspondence_retest
"""

import io
import os
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw

from tryon.avatar_representation import build_avatar_representation
from tryon.avatar_torso_landmarks_test import find_semantic_underarm
from tryon.correspondence_shoulders_test import find_garment_underarms
from tryon.garment_shoulder_region_extraction import (
    get_reference_orientation, walk_side, find_sustained_transition, get_binary_mask
)
import cv2 as cv2_mod

GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
OUTPUT_DIR = "outputs"


def fit_affine_least_squares(src_points, dst_points):
    src = np.array(src_points, dtype=np.float64)
    dst = np.array(dst_points, dtype=np.float64)
    n = len(src)
    A = np.zeros((2*n, 6))
    b = np.zeros(2*n)
    for i in range(n):
        x, y = src[i]
        A[2*i]   = [x, y, 0, 0, 1, 0]
        A[2*i+1] = [0, 0, x, y, 0, 1]
        b[2*i]   = dst[i][0]
        b[2*i+1] = dst[i][1]
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, bb, c, d, e, f = params
    return np.array([[a, bb, e], [c, d, f]])


def apply_matrix(matrix, pt):
    x, y = pt
    return (matrix[0,0]*x + matrix[0,1]*y + matrix[0,2],
            matrix[1,0]*x + matrix[1,1]*y + matrix[1,2])


def solve_similarity_2point(src_pts, dst_pts):
    s0, s1 = complex(*src_pts[0]), complex(*src_pts[1])
    d0, d1 = complex(*dst_pts[0]), complex(*dst_pts[1])
    a = (d1 - d0) / (s1 - s0)
    b = d0 - a * s0
    return np.array([[a.real, -a.imag, b.real], [a.imag, a.real, b.imag]])


def get_garment_shoulder_points(garment_mask, garment_img, left_ua, right_ua, gw):
    gray = cv2.cvtColor(np.array(garment_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    underarm_y = max(left_ua[1], right_ua[1])
    reference_angle = get_reference_orientation(gray, garment_mask, underarm_y, gw)

    left_walk = walk_side(gray, garment_mask, left_ua, reference_angle, gw, "left")
    right_walk = walk_side(gray, garment_mask, right_ua, reference_angle, gw, "right")

    left_transition = find_sustained_transition(left_walk)
    right_transition = find_sustained_transition(right_walk)

    left_shoulder = (left_transition[1], left_transition[0]) if left_transition else None
    right_shoulder = (right_transition[1], right_transition[0]) if right_transition else None
    return left_shoulder, right_shoulder


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Fetching golden garment...")
    garment_resp = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{GOLDEN_GARMENT_ID}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    garment_bytes = requests.get(garment_resp["clean_image_url"]).content
    garment_mask, garment_img = get_binary_mask(garment_bytes)
    gw, gh = garment_img.size

    g_left_ua, g_right_ua = find_garment_underarms(garment_mask, gw)
    g_left_sh, g_right_sh = get_garment_shoulder_points(garment_mask, garment_img, g_left_ua, g_right_ua, gw)
    print(f"Garment underarms: L={g_left_ua}  R={g_right_ua}")
    print(f"Garment shoulders (region-based): L={g_left_sh}  R={g_right_sh}")

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

    print(f"Avatar underarms: L={a_left_ua}  R={a_right_ua}")
    print(f"Avatar shoulders: L={rep.left_shoulder}  R={rep.right_shoulder}")

    # Pair by physical x-position (small-x <-> small-x)
    g_small_ua = g_left_ua if g_left_ua[0] < g_right_ua[0] else g_right_ua
    g_large_ua = g_right_ua if g_left_ua[0] < g_right_ua[0] else g_left_ua
    g_small_sh = g_left_sh if g_left_sh[0] < g_right_sh[0] else g_right_sh
    g_large_sh = g_right_sh if g_left_sh[0] < g_right_sh[0] else g_left_sh

    a_small_ua = a_left_ua if a_left_ua[0] < a_right_ua[0] else a_right_ua
    a_large_ua = a_right_ua if a_left_ua[0] < a_right_ua[0] else a_left_ua
    a_small_sh = rep.left_shoulder if rep.left_shoulder[0] < rep.right_shoulder[0] else rep.right_shoulder
    a_large_sh = rep.right_shoulder if rep.left_shoulder[0] < rep.right_shoulder[0] else rep.left_shoulder

    # ================= TEST 1: GLOBAL AFFINE =================
    src_points = [g_small_ua, g_large_ua, g_small_sh, g_large_sh]
    dst_points = [a_small_ua, a_large_ua, a_small_sh, a_large_sh]

    affine_matrix = fit_affine_least_squares(src_points, dst_points)
    print(f"\n=== GLOBAL AFFINE (re-test) ===")
    total_error = 0
    for name, src, dst in zip(["L_UA","R_UA","L_shoulder","R_shoulder"], src_points, dst_points):
        transformed = apply_matrix(affine_matrix, src)
        error = np.hypot(transformed[0]-dst[0], transformed[1]-dst[1])
        total_error += error
        print(f"  {name}: transformed=({transformed[0]:.1f},{transformed[1]:.1f})  target={dst}  error={error:.1f}px")
    print(f"  TOTAL residual error: {total_error:.1f}px")

    warped_affine = cv2.warpAffine(np.array(garment_img), affine_matrix, (aw, ah),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    result_affine = rep.image.convert("RGBA").copy()
    result_affine.paste(Image.fromarray(warped_affine), (0,0), Image.fromarray(warped_affine))
    result_affine.convert("RGB").save(f"{OUTPUT_DIR}/retest_affine.png")
    print(f"  Saved: {OUTPUT_DIR}/retest_affine.png")

    # ================= TEST 2: TWO-ZONE INDEPENDENT =================
    torso_matrix = solve_similarity_2point([g_small_ua, g_large_ua], [a_small_ua, a_large_ua])
    shoulder_matrix = solve_similarity_2point([g_small_sh, g_large_sh], [a_small_sh, a_large_sh])

    torso_scale = np.hypot(torso_matrix[0,0], torso_matrix[1,0])
    shoulder_scale = np.hypot(shoulder_matrix[0,0], shoulder_matrix[1,0])
    print(f"\n=== TWO-ZONE INDEPENDENT (re-test) ===")
    print(f"  Torso zone scale: {torso_scale:.4f}")
    print(f"  Shoulder zone scale: {shoulder_scale:.4f}")
    print(f"  Scale ratio: {shoulder_scale/torso_scale:.3f}  (previous obsolete result was ~2.05)")

    split_y = (g_small_ua[1] + g_large_ua[1]) // 2
    garment_arr = np.array(garment_img)
    shoulder_zone = garment_arr.copy(); shoulder_zone[split_y:, :, 3] = 0
    torso_zone = garment_arr.copy(); torso_zone[:split_y, :, 3] = 0

    warped_shoulder = cv2.warpAffine(shoulder_zone, shoulder_matrix, (aw, ah),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    warped_torso = cv2.warpAffine(torso_zone, torso_matrix, (aw, ah),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

    result_arr = np.array(rep.image.convert("RGBA")).astype(np.float32)
    for layer in [warped_torso, warped_shoulder]:
        alpha = layer[:,:,3:4].astype(np.float32) / 255.0
        result_arr = result_arr * (1-alpha) + layer.astype(np.float32) * alpha
    result_twozone = Image.fromarray(result_arr.astype(np.uint8))

    seam_check_torso = apply_matrix(torso_matrix, g_small_ua)
    seam_check_shoulder = apply_matrix(shoulder_matrix, (g_small_ua[0], split_y))
    seam_gap = np.hypot(seam_check_torso[0]-seam_check_shoulder[0], seam_check_torso[1]-seam_check_shoulder[1])
    print(f"  Seam gap at split (re-test): {seam_gap:.1f}px  (previous obsolete result was 93.1px)")

    result_twozone.convert("RGB").save(f"{OUTPUT_DIR}/retest_twozone.png")
    print(f"  Saved: {OUTPUT_DIR}/retest_twozone.png")

    print(f"\n>>> DECISION GATE: with the corrected shoulder landmark, does "
          f"either model now produce a visually credible, low-error result "
          f"— or does the mismatch persist even with valid correspondences?")


if __name__ == "__main__":
    main()