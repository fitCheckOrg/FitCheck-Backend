"""
Continuous shared-boundary piecewise deformation. Same 4 validated
anchors (2 shoulder + 2 underarm) as the affine/two-zone re-test,
but built as a proper triangle mesh with SHARED boundary vertices —
not two independently-solved zones. No hip anchors. Lower garment
(below underarm) is left untouched — governed by the already-
validated underarm-based similarity transform only, per explicit
scope.

Run: python -m tryon.piecewise_continuous_test
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

GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
OUTPUT_DIR = "outputs"


def solve_similarity_2point(src_pts, dst_pts):
    s0, s1 = complex(*src_pts[0]), complex(*src_pts[1])
    d0, d1 = complex(*dst_pts[0]), complex(*dst_pts[1])
    a = (d1 - d0) / (s1 - s0)
    b = d0 - a * s0
    return np.array([[a.real, -a.imag, b.real], [a.imag, a.real, b.imag]])


def warp_triangle(src_img, src_tri_pts, dst_tri_pts, output_size):
    src_tri = np.array(src_tri_pts, dtype=np.float32).reshape(3,2)
    dst_tri = np.array(dst_tri_pts, dtype=np.float32).reshape(3,2)
    src_rect = cv2.boundingRect(np.array(src_tri_pts, dtype=np.int32))
    dst_rect = cv2.boundingRect(np.array(dst_tri_pts, dtype=np.int32))
    if src_rect[2]<=0 or src_rect[3]<=0 or dst_rect[2]<=0 or dst_rect[3]<=0:
        return None, dst_rect
    src_off = np.array([[p[0]-src_rect[0], p[1]-src_rect[1]] for p in src_tri_pts], dtype=np.float32).reshape(3,2)
    dst_off = np.array([[p[0]-dst_rect[0], p[1]-dst_rect[1]] for p in dst_tri_pts], dtype=np.float32).reshape(3,2)
    src_crop = src_img[src_rect[1]:src_rect[1]+src_rect[3], src_rect[0]:src_rect[0]+src_rect[2]]
    if src_crop.size == 0:
        return None, dst_rect
    warp_mat = cv2.getAffineTransform(src_off, dst_off)
    dst_crop = cv2.warpAffine(src_crop, warp_mat, (dst_rect[2], dst_rect[3]), borderMode=cv2.BORDER_REFLECT_101)
    mask = np.zeros((dst_rect[3], dst_rect[2], 4), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_off.astype(np.int32), (255,255,255,255))
    dst_crop = cv2.bitwise_and(dst_crop, mask)
    return dst_crop, dst_rect


def composite_triangle(output, warped_crop, dst_rect):
    if warped_crop is None:
        return
    x, y, w, h = dst_rect
    x_end, y_end = min(x+w, output.shape[1]), min(y+h, output.shape[0])
    x_start, y_start = max(x,0), max(y,0)
    if x_end<=x_start or y_end<=y_start:
        return
    cx, cy = x_start-x, y_start-y
    region = output[y_start:y_end, x_start:x_end]
    wregion = warped_crop[cy:cy+(y_end-y_start), cx:cx+(x_end-x_start)]
    alpha = wregion[:,:,3:4].astype(np.float32)/255.0
    output[y_start:y_end, x_start:x_end] = (region*(1-alpha) + wregion.astype(np.float32)*alpha).astype(np.uint8)


def get_garment_shoulder_points(garment_mask, garment_img, left_ua, right_ua, gw):
    gray = cv2.cvtColor(np.array(garment_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    underarm_y = max(left_ua[1], right_ua[1])
    reference_angle = get_reference_orientation(gray, garment_mask, underarm_y, gw)
    left_walk = walk_side(gray, garment_mask, left_ua, reference_angle, gw, "left")
    right_walk = walk_side(gray, garment_mask, right_ua, reference_angle, gw, "right")
    lt = find_sustained_transition(left_walk)
    rt = find_sustained_transition(right_walk)
    return (lt[1], lt[0]) if lt else None, (rt[1], rt[0]) if rt else None


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
    print(f"Garment underarms: L={g_left_ua}  R={g_right_ua}")
    print(f"Garment shoulders: L={g_left_sh}  R={g_right_sh}")

    print("\nBuilding AvatarRepresentation...")
    rep = build_avatar_representation()
    aw, ah = rep.image.size

    a_l_uy, _ = find_semantic_underarm(rep.torso_mask, rep.upper_arms_mask, aw, "left")
    a_r_uy, _ = find_semantic_underarm(rep.torso_mask, rep.upper_arms_mask, aw, "right")
    a_l_row = rep.torso_mask[a_l_uy]; a_l_xs = np.where(a_l_row[:aw//2])[0]
    a_left_ua = (int(a_l_xs.min()), a_l_uy)
    a_r_row = rep.torso_mask[a_r_uy]; a_r_xs = np.where(a_r_row[aw//2:])[0] + aw//2
    a_right_ua = (int(a_r_xs.max()), a_r_uy)

    print(f"Avatar underarms: L={a_left_ua}  R={a_right_ua}")
    print(f"Avatar shoulders: L={rep.left_shoulder}  R={rep.right_shoulder}")

    # Pair by physical x-position
    g_s_ua = g_left_ua if g_left_ua[0]<g_right_ua[0] else g_right_ua
    g_l_ua = g_right_ua if g_left_ua[0]<g_right_ua[0] else g_left_ua
    g_s_sh = g_left_sh if g_left_sh[0]<g_right_sh[0] else g_right_sh
    g_l_sh = g_right_sh if g_left_sh[0]<g_right_sh[0] else g_left_sh
    a_s_ua = a_left_ua if a_left_ua[0]<a_right_ua[0] else a_right_ua
    a_l_ua = a_right_ua if a_left_ua[0]<a_right_ua[0] else a_left_ua
    a_s_sh = rep.left_shoulder if rep.left_shoulder[0]<rep.right_shoulder[0] else rep.right_shoulder
    a_l_sh = rep.right_shoulder if rep.left_shoulder[0]<rep.right_shoulder[0] else rep.left_shoulder

    # ===== SHARED-BOUNDARY MESH: upper zone = 2 triangles using 
    # shoulder + underarm vertices, sharing the underarm edge exactly =====
    # Triangle 1: S_small, S_large, UA_small
    # Triangle 2: S_large, UA_small, UA_large
    src_tris = [
        [g_s_sh, g_l_sh, g_s_ua],
        [g_l_sh, g_s_ua, g_l_ua],
    ]
    dst_tris = [
        [a_s_sh, a_l_sh, a_s_ua],
        [a_l_sh, a_s_ua, a_l_ua],
    ]

    output = np.array(rep.image.convert("RGBA")).astype(np.uint8)
    for src_tri, dst_tri in zip(src_tris, dst_tris):
        warped, rect = warp_triangle(garment_arr, src_tri, dst_tri, (aw, ah))
        composite_triangle(output, warped, rect)

    # ===== LOWER ZONE: unchanged, governed by existing underarm-based 
    # similarity transform, per explicit scope (no hip anchors) =====
    torso_matrix = solve_similarity_2point([g_s_ua, g_l_ua], [a_s_ua, a_l_ua])

    torso_zone = garment_arr.copy()
    x1, y1 = g_s_ua
    x2, y2 = g_l_ua
    yy, xx = np.indices(garment_mask.shape)
    boundary_y = y1 + (xx - x1) * (y2 - y1) / (x2 - x1)
    lower_keep = yy >= boundary_y
    torso_zone[:, :, 3] = np.where(lower_keep, torso_zone[:, :, 3], 0)
    
    warped_torso = cv2.warpAffine(torso_zone, torso_matrix, (aw, ah),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    alpha = warped_torso[:,:,3:4].astype(np.float32)/255.0
    output = (output.astype(np.float32)*(1-alpha) + warped_torso.astype(np.float32)*alpha).astype(np.uint8)

    result = Image.fromarray(output)

    draw = ImageDraw.Draw(result)
    for pt in [a_s_sh, a_l_sh, a_s_ua, a_l_ua]:
        x, y = pt
        draw.ellipse([x-5,y-5,x+5,y+5], outline=(0,255,0), width=2)

    result.convert("RGB").save(f"{OUTPUT_DIR}/piecewise_continuous_test.png")
    print(f"\nSaved: {OUTPUT_DIR}/piecewise_continuous_test.png")
    print(f"\n>>> DECISION GATE: is the shoulder-to-underarm region now "
          f"continuous (no seam), while remaining visually coherent — "
          f"i.e. does shared-boundary triangulation solve what independent "
          f"two-zone transforms could not?")


if __name__ == "__main__":
    main()