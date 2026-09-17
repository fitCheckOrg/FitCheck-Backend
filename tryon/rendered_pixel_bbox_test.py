"""
Direct measurement of actual rendered pixels. Uses the real pipeline
functions throughout, with the pre-allocated dst buffer fix for
cv2.warpAffine + BORDER_TRANSPARENT (uninitialized-memory bug found
mid-session — likely the same root cause as fresh_7's visual noise
artifact from the earlier review).

Measures the TORSO LAYER IN ISOLATION, before compositing onto the
mesh — the mesh layer is anchored correctly for every garment by
construction, so compositing on top of it masks whatever the torso
layer (the actually-unbounded, suspect layer) is doing.

Run: python -m tryon.rendered_pixel_bbox_test
"""

import os
import numpy as np
import cv2
import requests

from tryon.avatar_representation import build_avatar_representation
from tryon.avatar_torso_landmarks_test import find_semantic_underarm
from tryon.correspondence_shoulders_test import find_garment_underarms
from tryon.garment_shoulder_region_extraction import get_binary_mask
from tryon.piecewise_continuous_test import (
    get_garment_shoulder_points, warp_triangle, composite_triangle,
    solve_similarity_2point
)

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
OUTPUT_DIR = "outputs"

GARMENTS = [
    {"id": "ac7fffb7-2e3a-40f8-894a-0e85fc245373", "label": "golden_polo_control", "verdict": "PASS"},
    {"id": "5c22b2ea-fedc-4dbb-b649-fc645774d011", "label": "fresh_3_t-shirt", "verdict": "FAIL"},
    {"id": "42b7a010-c4fa-4b8f-8ed2-10c3088967b5", "label": "fresh_8_polo", "verdict": "PASS"},
]


def render_and_measure(rep, garment_id):
    aw, ah = rep.image.size

    garment_resp = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{garment_id}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    garment_bytes = requests.get(garment_resp["clean_image_url"]).content
    garment_mask, garment_img = get_binary_mask(garment_bytes)
    gw, gh = garment_img.size
    garment_arr = np.array(garment_img)

    g_left_ua, g_right_ua = find_garment_underarms(garment_mask, gw)
    g_left_sh, g_right_sh = get_garment_shoulder_points(garment_mask, garment_img, g_left_ua, g_right_ua, gw)

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

    # Mesh layer (bounded, always correctly anchored — not the suspect)
    src_tris = [[g_s_sh, g_l_sh, g_s_ua], [g_l_sh, g_s_ua, g_l_ua]]
    dst_tris = [[a_s_sh, a_l_sh, a_s_ua], [a_l_sh, a_s_ua, a_l_ua]]
    mesh_layer = np.zeros((ah, aw, 4), dtype=np.uint8)
    for src_tri, dst_tri in zip(src_tris, dst_tris):
        warped, rect = warp_triangle(garment_arr, src_tri, dst_tri, (aw, ah))
        composite_triangle(mesh_layer, warped, rect)

    # Torso layer (unbounded warpAffine — the actual suspect)
    torso_matrix = solve_similarity_2point([g_s_ua, g_l_ua], [a_s_ua, a_l_ua])
    x1, y1 = g_s_ua
    x2, y2 = g_l_ua
    yy, xx = np.indices(garment_mask.shape)
    boundary_y = y1 + (xx - x1) * (y2 - y1) / (x2 - x1)
    lower_keep = yy >= boundary_y
    torso_zone = garment_arr.copy()
    torso_zone[:, :, 3] = np.where(lower_keep, torso_zone[:, :, 3], 0)

    dst_buffer = np.zeros((ah, aw, 4), dtype=np.uint8)  # pre-allocated, zero-filled — fixes the uninitialized-memory bug
    warped_torso = cv2.warpAffine(torso_zone, torso_matrix, (aw, ah), dst=dst_buffer,
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

    # Measure the TORSO LAYER ALONE, before compositing onto the mesh
    torso_alpha = warped_torso[:, :, 3]
    t_ys, t_xs = np.where(torso_alpha > 10)
    if len(t_ys) == 0:
        torso_metrics = {"torso_top_y_norm": None, "torso_bottom_y_norm": None,
                          "torso_left_x": None, "torso_right_x": None, "torso_pixel_count": 0}
    else:
        torso_metrics = {
            "torso_top_y_norm": t_ys.min() / ah,
            "torso_bottom_y_norm": t_ys.max() / ah,
            "torso_left_x": int(t_xs.min()),
            "torso_right_x": int(t_xs.max()),
            "torso_pixel_count": int(len(t_ys)),
        }

    # Also report the final composited result, for reference
    alpha_t = warped_torso[:, :, 3:4].astype(np.float32) / 255.0
    composited = (mesh_layer.astype(np.float32) * (1 - alpha_t) +
                   warped_torso.astype(np.float32) * alpha_t).astype(np.uint8)
    c_ys, c_xs = np.where(composited[:, :, 3] > 10)
    composite_metrics = {
        "composite_top_y_norm": c_ys.min() / ah if len(c_ys) else None,
        "composite_bottom_y_norm": c_ys.max() / ah if len(c_ys) else None,
    }

    return {
        **torso_metrics, **composite_metrics,
        "avatar_shoulder_y_norm": ((a_s_sh[1] + a_l_sh[1]) / 2) / ah,
        "avatar_underarm_y_norm": ((a_s_ua[1] + a_l_ua[1]) / 2) / ah,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Building AvatarRepresentation (shared across all garments)...")
    rep = build_avatar_representation()
    aw, ah = rep.image.size
    print(f"Avatar dimensions: {aw} x {ah}\n")

    print(f"{'Garment':<22}{'Verdict':<10}{'torso_top':<12}{'torso_bottom':<14}{'torso_L':<9}{'torso_R':<9}{'torso_px'}")
    print("-" * 90)

    for g in GARMENTS:
        m = render_and_measure(rep, g["id"])
        top_str = f"{m['torso_top_y_norm']:.3f}" if m['torso_top_y_norm'] is not None else "N/A"
        bot_str = f"{m['torso_bottom_y_norm']:.3f}" if m['torso_bottom_y_norm'] is not None else "N/A"
        left_str = f"{m['torso_left_x']}" if m['torso_left_x'] is not None else "N/A"
        right_str = f"{m['torso_right_x']}" if m['torso_right_x'] is not None else "N/A"
        print(f"{g['label']:<22}{g['verdict']:<10}{top_str:<12}{bot_str:<14}{left_str:<9}{right_str:<9}{m['torso_pixel_count']}")
        print(f"  composite: top={m['composite_top_y_norm']:.3f}  bottom={m['composite_bottom_y_norm']:.3f}  "
              f"(avatar shoulder={m['avatar_shoulder_y_norm']:.3f}, underarm={m['avatar_underarm_y_norm']:.3f})")


if __name__ == "__main__":
    main()