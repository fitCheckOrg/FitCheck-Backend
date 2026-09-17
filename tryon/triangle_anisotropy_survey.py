"""
Measures triangle-scale anisotropy (max(x_scale,y_scale) /
min(x_scale,y_scale)) for both mesh triangles, across all 7
already-reviewed in-contract garments. Pure measurement — reuses
the exact same landmark-extraction and triangle-construction logic
already in the pipeline, no new heuristics, no fitting changes.

Tests whether fresh_3 (and possibly fresh_4) are measurable
outliers on this axis, or whether anisotropy is more widespread
than the visual review alone suggested.

Run: python -m tryon.triangle_anisotropy_survey
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
OUTPUT_DIR = "outputs"

# The 7 confirmed in-contract garments from the visual review, 
# with their known overall verdict for direct comparison
GARMENTS = [
    {"id": "ac7fffb7-2e3a-40f8-894a-0e85fc245373", "label": "golden_polo_control", "verdict": "PASS"},
    {"id": "5c22b2ea-fedc-4dbb-b649-fc645774d011", "label": "fresh_3_t-shirt", "verdict": "FAIL"},
    {"id": "d5d05081-fdb8-4104-8b56-53bb06e2bc93", "label": "fresh_4_polo", "verdict": "MINOR_ARTIFACT"},
    {"id": "3a46afa7-e179-4d8b-a007-ef9af08fe27b", "label": "fresh_5_short-sleeve", "verdict": "PASS"},
    {"id": "d7256fcb-c37d-4a44-88de-d593ad61f20e", "label": "fresh_7_polo", "verdict": "FAIL(rendering)"},
    {"id": "42b7a010-c4fa-4b8f-8ed2-10c3088967b5", "label": "fresh_8_polo", "verdict": "PASS"},
    {"id": "831098a3-8778-480f-9a11-aa9df826d0f2", "label": "fresh_9_polo", "verdict": "PASS"},
]


def get_triangle_anisotropy(rep, garment_id):
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

    results = []
    for i, (src_tri, dst_tri) in enumerate(zip(src_tris, dst_tris)):
        src = np.array(src_tri, dtype=np.float32)
        dst = np.array(dst_tri, dtype=np.float32)
        matrix = cv2.getAffineTransform(src, dst)
        scale_x = np.hypot(matrix[0,0], matrix[1,0])
        scale_y = np.hypot(matrix[0,1], matrix[1,1])
        anisotropy = max(scale_x, scale_y) / min(scale_x, scale_y)
        results.append({"triangle": i+1, "scale_x": scale_x, "scale_y": scale_y, "anisotropy": anisotropy})

    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Building AvatarRepresentation (shared across all garments)...")
    rep = build_avatar_representation()

    print(f"\n{'Garment':<22}{'Verdict':<18}{'Tri':<5}{'scale_x':<10}{'scale_y':<10}{'anisotropy'}")
    print("-" * 80)

    all_results = []
    for g in GARMENTS:
        try:
            tris = get_triangle_anisotropy(rep, g["id"])
            for t in tris:
                print(f"{g['label']:<22}{g['verdict']:<18}{t['triangle']:<5}"
                      f"{t['scale_x']:<10.3f}{t['scale_y']:<10.3f}{t['anisotropy']:.2f}")
                all_results.append({**g, **t})
        except Exception as e:
            print(f"{g['label']:<22}{g['verdict']:<18} ERROR: {e}")

    print(f"\n--- Sorted by anisotropy (highest first) ---")
    for r in sorted(all_results, key=lambda x: -x["anisotropy"]):
        print(f"  {r['anisotropy']:.2f}x  {r['label']} (Triangle {r['triangle']}, verdict={r['verdict']})")

    print(f"\n>>> Look for: does a clear anisotropy threshold separate PASS "
          f"garments from fresh_3/fresh_4, or is the picture more mixed "
          f"than that?")


if __name__ == "__main__":
    main()