"""
Measures DESTINATION POSITION (not shape) for each triangle across
the same 7 garments, testing whether where the garment lands —
specifically the garment's own top edge, mapped through Triangle 1
— separates fresh_3 from the PASS cases where anisotropy and shear
both failed to.

Same exact landmarks, same exact transforms as the real pipeline.
Measurement only.

Run: python -m tryon.triangle_position_survey
"""

import os
import numpy as np
import cv2
import requests

from tryon.avatar_representation import build_avatar_representation
from tryon.avatar_torso_landmarks_test import find_semantic_underarm
from tryon.correspondence_shoulders_test import find_garment_underarms
from tryon.garment_shoulder_region_extraction import get_binary_mask
from tryon.piecewise_continuous_test import get_garment_shoulder_points

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
OUTPUT_DIR = "outputs"

GARMENTS = [
    {"id": "ac7fffb7-2e3a-40f8-894a-0e85fc245373", "label": "golden_polo_control", "verdict": "PASS"},
    {"id": "5c22b2ea-fedc-4dbb-b649-fc645774d011", "label": "fresh_3_t-shirt", "verdict": "FAIL"},
    {"id": "d5d05081-fdb8-4104-8b56-53bb06e2bc93", "label": "fresh_4_polo", "verdict": "MINOR_ARTIFACT"},
    {"id": "3a46afa7-e179-4d8b-a007-ef9af08fe27b", "label": "fresh_5_short-sleeve", "verdict": "PASS"},
    {"id": "d7256fcb-c37d-4a44-88de-d593ad61f20e", "label": "fresh_7_polo", "verdict": "FAIL(rendering)"},
    {"id": "42b7a010-c4fa-4b8f-8ed2-10c3088967b5", "label": "fresh_8_polo", "verdict": "PASS"},
    {"id": "831098a3-8778-480f-9a11-aa9df826d0f2", "label": "fresh_9_polo", "verdict": "PASS"},
]


def get_position_metrics(rep, garment_id):
    garment_resp = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{garment_id}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    garment_bytes = requests.get(garment_resp["clean_image_url"]).content
    garment_mask, garment_img = get_binary_mask(garment_bytes)
    gw, gh = garment_img.size
    aw, ah = rep.image.size

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

    src_tris = [[g_s_sh, g_l_sh, g_s_ua], [g_l_sh, g_s_ua, g_l_ua]]
    dst_tris = [[a_s_sh, a_l_sh, a_s_ua], [a_l_sh, a_s_ua, a_l_ua]]

    def apply_matrix(matrix, pt):
        x, y = pt
        return (matrix[0,0]*x + matrix[0,1]*y + matrix[0,2],
                matrix[1,0]*x + matrix[1,1]*y + matrix[1,2])

    # Triangle 1 (upper/shoulder zone) matrix, used for the garment's 
    # own top-edge points
    src1 = np.array(src_tris[0], dtype=np.float32)
    dst1 = np.array(dst_tris[0], dtype=np.float32)
    matrix1 = cv2.getAffineTransform(src1, dst1)

    # A. Triangle 1 centroid, source and destination
    src_centroid = np.mean(src_tris[0], axis=0)
    dst_centroid = np.mean(dst_tris[0], axis=0)

    # B. Garment's own top-edge points (y=0, spanning the source 
    # triangle's x-range), mapped through Triangle 1 -- the more 
    # diagnostic measurement per the explicit correction
    top_left_src = (min(p[0] for p in src_tris[0]), 0)
    top_right_src = (max(p[0] for p in src_tris[0]), 0)
    top_left_dst = apply_matrix(matrix1, top_left_src)
    top_right_dst = apply_matrix(matrix1, top_right_src)
    mapped_top_y = min(top_left_dst[1], top_right_dst[1])

    return {
        "src_centroid": tuple(src_centroid),
        "dst_centroid": tuple(dst_centroid),
        "dst_centroid_y_norm": dst_centroid[1] / ah,
        "mapped_top_y": mapped_top_y,
        "top_y_normalized": mapped_top_y / ah,
        "avatar_underarm_y_norm": ((a_s_ua[1] + a_l_ua[1]) / 2) / ah,
        "avatar_shoulder_y_norm": ((a_s_sh[1] + a_l_sh[1]) / 2) / ah,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Building AvatarRepresentation (shared across all garments)...")
    rep = build_avatar_representation()

    print(f"\n{'Garment':<22}{'Verdict':<18}{'dst_cent_y_norm':<18}{'top_y_norm':<14}{'UA_y_norm':<12}{'Sh_y_norm'}")
    print("-" * 100)

    all_results = []
    for g in GARMENTS:
        try:
            m = get_position_metrics(rep, g["id"])
            print(f"{g['label']:<22}{g['verdict']:<18}{m['dst_centroid_y_norm']:<18.3f}"
                  f"{m['top_y_normalized']:<14.3f}{m['avatar_underarm_y_norm']:<12.3f}"
                  f"{m['avatar_shoulder_y_norm']:.3f}")
            all_results.append({**g, **m})
        except Exception as e:
            print(f"{g['label']:<22}{g['verdict']:<18} ERROR: {e}")

    print(f"\n--- Sorted by top_y_normalized (lowest = highest on avatar, closest to head) ---")
    for r in sorted(all_results, key=lambda x: x["top_y_normalized"]):
        print(f"  top_y={r['top_y_normalized']:.3f}  centroid_y={r['dst_centroid_y_norm']:.3f}  "
              f"{r['label']} (verdict={r['verdict']})")

    print(f"\n>>> Look for: does fresh_3 show a distinctly lower top_y_normalized "
          f"(garment top mapped too high, toward the head) compared to the "
          f"PASS garments, or does the distribution overlap heavily?")


if __name__ == "__main__":
    main()