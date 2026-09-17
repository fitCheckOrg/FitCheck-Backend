"""
Compares two ways of handling the free-hanging lower garment, with
NO hip anchor in either:

BASELINE: current approach — a fresh similarity transform solved
    independently from underarm points only (what's been used all
    session).
CANDIDATE: extends the upper mesh's own affine field (the lower
    triangle's matrix, already anchored at both underarms) downward
    over the rest of the garment, rather than re-solving.

Both anchor identically at the underarms — the only question is
whether continuing the SAME local mapping vs. re-solving a fresh
one produces a meaningfully different, better, or worse hem result.

Run: python -m tryon.lower_body_deformation_field_test
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
from tryon.piecewise_continuous_test import (
    solve_similarity_2point, warp_triangle, composite_triangle, get_garment_shoulder_points
)

GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
OUTPUT_DIR = "outputs"


def get_affine_matrix_from_triangle(src_tri, dst_tri):
    """Returns the FULL 2x3 affine matrix mapping src_tri -> dst_tri
    exactly (3-point exact solve, standard for a single triangle)."""
    src = np.array(src_tri, dtype=np.float32)
    dst = np.array(dst_tri, dtype=np.float32)
    return cv2.getAffineTransform(src, dst)


def apply_matrix(matrix, pt):
    x, y = pt
    return (matrix[0,0]*x + matrix[0,1]*y + matrix[0,2],
            matrix[1,0]*x + matrix[1,1]*y + matrix[1,2])


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
    g_left_sh, g_right_sh = get_garment_shoulder_points(garment_mask, garment_img, g_left_ua, g_right_ua, gw)

    print("\nBuilding AvatarRepresentation...")
    rep = build_avatar_representation()
    aw, ah = rep.image.size

    a_l_uy, _ = find_semantic_underarm(rep.torso_mask, rep.upper_arms_mask, aw, "left")
    a_r_uy, _ = find_semantic_underarm(rep.torso_mask, rep.upper_arms_mask, aw, "right")
    a_l_row = rep.torso_mask[a_l_uy]; a_l_xs = np.where(a_l_row[:aw//2])[0]
    a_left_ua = (int(a_l_xs.min()), a_l_uy)
    a_r_row = rep.torso_mask[a_r_uy]; a_r_xs = np.where(a_r_row[aw//2:])[0] + aw//2
    a_right_ua = (int(a_r_xs.max()), a_r_uy)

    g_s_ua = g_left_ua if g_left_ua[0]<g_right_ua[0] else g_right_ua
    g_l_ua = g_right_ua if g_left_ua[0]<g_right_ua[0] else g_left_ua
    g_s_sh = g_left_sh if g_left_sh[0]<g_right_sh[0] else g_right_sh
    g_l_sh = g_right_sh if g_left_sh[0]<g_right_sh[0] else g_left_sh
    a_s_ua = a_left_ua if a_left_ua[0]<a_right_ua[0] else a_right_ua
    a_l_ua = a_right_ua if a_left_ua[0]<a_right_ua[0] else a_left_ua
    a_s_sh = rep.left_shoulder if rep.left_shoulder[0]<rep.right_shoulder[0] else rep.right_shoulder
    a_l_sh = rep.right_shoulder if rep.left_shoulder[0]<rep.right_shoulder[0] else rep.left_shoulder

    # Upper mesh (unchanged, validated)
    src_tris = [[g_s_sh, g_l_sh, g_s_ua], [g_l_sh, g_s_ua, g_l_ua]]
    dst_tris = [[a_s_sh, a_l_sh, a_s_ua], [a_l_sh, a_s_ua, a_l_ua]]
    output = np.array(rep.image.convert("RGBA")).astype(np.uint8)
    for src_tri, dst_tri in zip(src_tris, dst_tris):
        warped, rect = warp_triangle(garment_arr, src_tri, dst_tri, (aw, ah))
        composite_triangle(output, warped, rect)

    # ===== BASELINE: fresh similarity transform (current approach) =====
    baseline_matrix = solve_similarity_2point([g_s_ua, g_l_ua], [a_s_ua, a_l_ua])
    x1, y1 = g_s_ua; x2, y2 = g_l_ua
    yy, xx = np.indices(garment_mask.shape)
    boundary_y = y1 + (xx - x1) * (y2 - y1) / (x2 - x1)
    lower_keep = yy >= boundary_y
    torso_zone_baseline = garment_arr.copy()
    torso_zone_baseline[:,:,3] = np.where(lower_keep, torso_zone_baseline[:,:,3], 0)
    warped_baseline = cv2.warpAffine(torso_zone_baseline, baseline_matrix, (aw, ah),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

    # ===== CANDIDATE: extend the lower triangle's OWN affine field =====
    # (the triangle [g_l_sh, g_s_ua, g_l_ua] -> [a_l_sh, a_s_ua, a_l_ua] 
    # already anchored at both underarms — reuse its exact matrix)
    field_matrix = get_affine_matrix_from_triangle(
        [g_l_sh, g_s_ua, g_l_ua], [a_l_sh, a_s_ua, a_l_ua]
    )
    torso_zone_field = garment_arr.copy()
    torso_zone_field[:,:,3] = np.where(lower_keep, torso_zone_field[:,:,3], 0)
    warped_field = cv2.warpAffine(torso_zone_field, field_matrix, (aw, ah),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

    # Composite both variants for side-by-side comparison
    result_baseline = output.copy()
    alpha_b = warped_baseline[:,:,3:4].astype(np.float32)/255.0
    result_baseline = (result_baseline.astype(np.float32)*(1-alpha_b) + warped_baseline.astype(np.float32)*alpha_b).astype(np.uint8)

    result_field = output.copy()
    alpha_f = warped_field[:,:,3:4].astype(np.float32)/255.0
    result_field = (result_field.astype(np.float32)*(1-alpha_f) + warped_field.astype(np.float32)*alpha_f).astype(np.uint8)

    Image.fromarray(result_baseline).convert("RGB").save(f"{OUTPUT_DIR}/lower_baseline_similarity.png")
    Image.fromarray(result_field).convert("RGB").save(f"{OUTPUT_DIR}/lower_field_extension.png")

    # Report hem endpoint position for both, at the garment's own hem y
    garment_hem_y = int(max(garment_mask.nonzero()[0]))
    hem_left_garment = (g_s_ua[0], garment_hem_y)
    hem_right_garment = (g_l_ua[0], garment_hem_y)

    hem_left_baseline = apply_matrix(baseline_matrix, hem_left_garment)
    hem_right_baseline = apply_matrix(baseline_matrix, hem_right_garment)
    hem_left_field = apply_matrix(field_matrix, hem_left_garment)
    hem_right_field = apply_matrix(field_matrix, hem_right_garment)

    print(f"\nGarment hem (source): L={hem_left_garment}  R={hem_right_garment}")
    print(f"\nBASELINE hem destination:  L=({hem_left_baseline[0]:.1f},{hem_left_baseline[1]:.1f})  "
          f"R=({hem_right_baseline[0]:.1f},{hem_right_baseline[1]:.1f})")
    print(f"FIELD-EXTENSION hem destination: L=({hem_left_field[0]:.1f},{hem_left_field[1]:.1f})  "
          f"R=({hem_right_field[0]:.1f},{hem_right_field[1]:.1f})")

    diff_left = np.hypot(hem_left_baseline[0]-hem_left_field[0], hem_left_baseline[1]-hem_left_field[1])
    diff_right = np.hypot(hem_right_baseline[0]-hem_right_field[0], hem_right_baseline[1]-hem_right_field[1])
    print(f"\nHem endpoint difference between the two methods: L={diff_left:.1f}px  R={diff_right:.1f}px")

    print(f"\nSaved: {OUTPUT_DIR}/lower_baseline_similarity.png")
    print(f"Saved: {OUTPUT_DIR}/lower_field_extension.png")
    print(f"\n>>> DECISION GATE: do the two methods produce meaningfully "
          f"different hems, or are they close enough that the simpler "
          f"baseline (already in use) is sufficient — meaning V1 genuinely "
          f"doesn't need a body-side hem landmark at all?")


if __name__ == "__main__":
    main()