"""
Isolates the cause of the faint dashed-line artifact at the upper-
mesh/lower-similarity boundary. Tests three things separately:

A. Geometry continuity — do the upper mesh's bottom-edge pixels and
   the lower similarity transform's top-edge pixels land at the
   SAME coordinates?
B. Texture continuity — do the shirt stripes align in color/position
   across the boundary, or is there a visible jump?
C. Alpha/compositing — is the dashed line a rendering artifact from
   how the two layers are blended (e.g. semi-transparent edge
   pixels from anti-aliasing), rather than a geometric gap?

No deformation changes. No new anchors. Pure diagnostic.

Run: python -m tryon.lower_region_continuity_test
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

    # ===== A. GEOMETRY CHECK =====
    # Where does the UPPER mesh's bottom edge (the underarm vertices, 
    # already shared) actually land? And separately, where does the 
    # LOWER similarity transform's TOP edge (a point at the same 
    # garment y as the underarm) land? These should be IDENTICAL 
    # since both use the same underarm anchors — but let's verify 
    # rather than assume.
    torso_matrix = solve_similarity_2point([g_s_ua, g_l_ua], [a_s_ua, a_l_ua])

    def apply_matrix(matrix, pt):
        x, y = pt
        return (matrix[0,0]*x + matrix[0,1]*y + matrix[0,2],
                matrix[1,0]*x + matrix[1,1]*y + matrix[1,2])

    lower_transform_check_L = apply_matrix(torso_matrix, g_s_ua)
    lower_transform_check_R = apply_matrix(torso_matrix, g_l_ua)

    print(f"\n=== A. GEOMETRY CONTINUITY ===")
    print(f"  Upper mesh bottom-left vertex (exact anchor): {a_s_ua}")
    print(f"  Lower transform's mapped underarm-left:       ({lower_transform_check_L[0]:.2f}, {lower_transform_check_L[1]:.2f})")
    print(f"  Gap: {np.hypot(a_s_ua[0]-lower_transform_check_L[0], a_s_ua[1]-lower_transform_check_L[1]):.3f}px")
    print(f"  Upper mesh bottom-right vertex (exact anchor): {a_l_ua}")
    print(f"  Lower transform's mapped underarm-right:       ({lower_transform_check_R[0]:.2f}, {lower_transform_check_R[1]:.2f})")
    print(f"  Gap: {np.hypot(a_l_ua[0]-lower_transform_check_R[0], a_l_ua[1]-lower_transform_check_R[1]):.3f}px")

    # ===== Render each layer SEPARATELY (not composited) to inspect 
    # each in isolation, plus a zoomed composite crop around the seam =====
    src_tris = [[g_s_sh, g_l_sh, g_s_ua], [g_l_sh, g_s_ua, g_l_ua]]
    dst_tris = [[a_s_sh, a_l_sh, a_s_ua], [a_l_sh, a_s_ua, a_l_ua]]

    upper_layer = np.zeros((ah, aw, 4), dtype=np.uint8)
    for src_tri, dst_tri in zip(src_tris, dst_tris):
        warped, rect = warp_triangle(garment_arr, src_tri, dst_tri, (aw, ah))
        composite_triangle(upper_layer, warped, rect)

    torso_zone = garment_arr.copy()
    x1, y1 = g_s_ua
    x2, y2 = g_l_ua
    yy, xx = np.indices(garment_mask.shape)
    boundary_y = y1 + (xx - x1) * (y2 - y1) / (x2 - x1)
    lower_keep = yy >= boundary_y
    torso_zone[:, :, 3] = np.where(lower_keep, torso_zone[:, :, 3], 0)
    
    lower_layer = cv2.warpAffine(torso_zone, torso_matrix, (aw, ah),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

    Image.fromarray(upper_layer).save(f"{OUTPUT_DIR}/layer_upper_only.png")
    Image.fromarray(lower_layer).save(f"{OUTPUT_DIR}/layer_lower_only.png")
    print(f"\nSaved isolated layers: layer_upper_only.png, layer_lower_only.png")

    # ===== C. ALPHA CHECK: sample alpha values along a horizontal 
    # line right at the seam, for BOTH layers =====
    seam_y = int(a_s_ua[1])
    upper_alpha_row = upper_layer[seam_y-2:seam_y+3, :, 3]
    lower_alpha_row = lower_layer[seam_y-2:seam_y+3, :, 3]
    print(f"\n=== C. ALPHA AT SEAM (y={seam_y}, x-range with any signal) ===")
    upper_nonzero = np.where(upper_alpha_row.max(axis=0) > 0)[0]
    lower_nonzero = np.where(lower_alpha_row.max(axis=0) > 0)[0]
    if len(upper_nonzero) and len(lower_nonzero):
        print(f"  Upper layer alpha range at seam: x=[{upper_nonzero.min()}, {upper_nonzero.max()}], "
              f"mean_alpha={upper_alpha_row[upper_alpha_row>0].mean():.1f}")
        print(f"  Lower layer alpha range at seam: x=[{lower_nonzero.min()}, {lower_nonzero.max()}], "
              f"mean_alpha={lower_alpha_row[lower_alpha_row>0].mean():.1f}")

    # Zoomed crop of the composited result around the seam, for visual B/C check
    composite = np.array(rep.image.convert("RGBA")).astype(np.float32)
    for layer in [lower_layer, upper_layer]:
        alpha = layer[:,:,3:4].astype(np.float32)/255.0
        composite = composite*(1-alpha) + layer.astype(np.float32)*alpha
    composite_img = Image.fromarray(composite.astype(np.uint8))

    crop_box = (int(a_s_ua[0])-15, seam_y-25, int(a_l_ua[0])+15, seam_y+25)
    zoomed = composite_img.crop(crop_box)
    zoomed = zoomed.resize((zoomed.width*4, zoomed.height*4), Image.NEAREST)
    zoomed.convert("RGB").save(f"{OUTPUT_DIR}/seam_zoom.png")
    print(f"\nSaved: {OUTPUT_DIR}/seam_zoom.png (4x zoom, NEAREST — no smoothing, so real pixels are visible)")

    print(f"\n>>> DECISION GATE: (A) is the geometry gap ~0px (confirming exact "
          f"coordinate match)? (B) do stripes align across the seam in the "
          f"zoomed crop? (C) does the alpha sampling show a gap/dip at the "
          f"seam row, or full coverage — pointing to compositing rather than "
          f"geometry as the cause?")


if __name__ == "__main__":
    main()