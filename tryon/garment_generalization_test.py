"""
V1 garment generalization test. Runs the EXACT SAME pipeline
(underarm detection -> region-orientation shoulder detection ->
4-point correspondence -> shared-boundary mesh -> underarm-anchored
lower similarity -> render) across multiple garments, unchanged.

No per-garment tuning. Records which stage fails, if any, rather
than patching around failures.

Run: python -m tryon.garment_generalization_test
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
from tryon.garment_shoulder_region_extraction import (
    get_reference_orientation, walk_side, find_sustained_transition, get_binary_mask
)
from tryon.piecewise_continuous_test import (
    solve_similarity_2point, warp_triangle, composite_triangle, get_garment_shoulder_points
)

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
OUTPUT_DIR = "outputs"

# Development items excluded from auto-discovery (already used, not 
# fresh test garments)
DEVELOPMENT_ITEM_IDS = {
    "ac7fffb7-2e3a-40f8-894a-0e85fc245373",  # golden polo
    "25989827-db9d-4d65-90b3-12f5357b0b44",
    "2ea4bb57-a358-4349-8aaf-204ded0772ab",
    "6a53c00f-4f84-4e64-ad3b-5d2753504481",
    "bb7763f6-f1df-4b26-9522-94299149bb5e",
    "f2944c50-5757-4abc-81a6-c1380f8419a0",
    "d20577a1-01e9-4d6f-87a1-5ca89c3badba",
    "52f5fd29-d817-49bb-a0c8-ba181ba47157",
    "cb13467e-585f-4a4f-82fe-bac29082424e",
    "ded59bd3-57a1-43b1-888a-867e4362d2d8",
    "c93a3138-f40f-4e91-8d91-be3ff1c3065f",
    "5d19a29d-a7fa-4579-a1ba-2e374c3fd410",
    "fe419f35-3b62-43b9-842c-8883651e9c4f",
    "62930780-680d-4653-a8fa-daa7a6900617",
    "bd85602f-8c93-4d2c-99e9-7eafdb6da425",
    "2a32b5bb-c3ea-48f9-ac8d-5849eee866ba",
    "e398e450-a4be-447a-b981-5a57297c7e66",
    "ba2e5aa1-bfb5-462f-aaad-fe378f658c42",
    "729ddf0c-fff0-4d4e-ae58-8bd32ddad32a",
}
GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"


def run_pipeline_for_garment(garment_id, garment_label, rep, aw, ah):
    """Runs the full unchanged pipeline for one garment against the
    already-built avatar representation. Returns a dict recording
    which stage succeeded/failed and why."""
    result = {"garment_id": garment_id, "label": garment_label, "stage_reached": None, "error": None}

    try:
        garment_resp = requests.get(
            f"http://127.0.0.1:8000/api/wardrobe/{garment_id}?user_id={AVATAR_USER_ID}"
        ).json()["data"]
        garment_bytes = requests.get(garment_resp["clean_image_url"]).content
        garment_mask, garment_img = get_binary_mask(garment_bytes)
        gw, gh = garment_img.size
        garment_arr = np.array(garment_img)
        result["stage_reached"] = "mask_loaded"
    except Exception as e:
        result["error"] = f"mask_loading: {e}"
        return result

    try:
        g_left_ua, g_right_ua = find_garment_underarms(garment_mask, gw)
        result["underarms"] = {"L": g_left_ua, "R": g_right_ua}
        result["stage_reached"] = "underarms_found"
    except Exception as e:
        result["error"] = f"underarm_detection: {e}"
        return result

    try:
        g_left_sh, g_right_sh = get_garment_shoulder_points(garment_mask, garment_img, g_left_ua, g_right_ua, gw)
        if g_left_sh is None or g_right_sh is None:
            result["error"] = "shoulder_detection: no sustained transition found on one or both sides"
            return result
        result["shoulders"] = {"L": g_left_sh, "R": g_right_sh}
        result["stage_reached"] = "shoulders_found"
    except Exception as e:
        result["error"] = f"shoulder_detection: {e}"
        return result

    try:
        g_s_ua = g_left_ua if g_left_ua[0]<g_right_ua[0] else g_right_ua
        g_l_ua = g_right_ua if g_left_ua[0]<g_right_ua[0] else g_left_ua
        g_s_sh = g_left_sh if g_left_sh[0]<g_right_sh[0] else g_right_sh
        g_l_sh = g_right_sh if g_left_sh[0]<g_right_sh[0] else g_left_sh

        a_l_uy, _ = find_semantic_underarm(rep.torso_mask, rep.upper_arms_mask, aw, "left")
        a_r_uy, _ = find_semantic_underarm(rep.torso_mask, rep.upper_arms_mask, aw, "right")
        a_l_row = rep.torso_mask[a_l_uy]; a_l_xs = np.where(a_l_row[:aw//2])[0]
        a_left_ua = (int(a_l_xs.min()), a_l_uy)
        a_r_row = rep.torso_mask[a_r_uy]; a_r_xs = np.where(a_r_row[aw//2:])[0] + aw//2
        a_right_ua = (int(a_r_xs.max()), a_r_uy)

        a_s_ua = a_left_ua if a_left_ua[0]<a_right_ua[0] else a_right_ua
        a_l_ua = a_right_ua if a_left_ua[0]<a_right_ua[0] else a_left_ua
        a_s_sh = rep.left_shoulder if rep.left_shoulder[0]<rep.right_shoulder[0] else rep.right_shoulder
        a_l_sh = rep.right_shoulder if rep.left_shoulder[0]<rep.right_shoulder[0] else rep.left_shoulder

        src_tris = [[g_s_sh, g_l_sh, g_s_ua], [g_l_sh, g_s_ua, g_l_ua]]
        dst_tris = [[a_s_sh, a_l_sh, a_s_ua], [a_l_sh, a_s_ua, a_l_ua]]

        output = np.array(rep.image.convert("RGBA")).astype(np.uint8)
        for src_tri, dst_tri in zip(src_tris, dst_tris):
            warped, rect = warp_triangle(garment_arr, src_tri, dst_tri, (aw, ah))
            composite_triangle(output, warped, rect)
        result["stage_reached"] = "mesh_rendered"

        torso_matrix = solve_similarity_2point([g_s_ua, g_l_ua], [a_s_ua, a_l_ua])
        x1, y1 = g_s_ua; x2, y2 = g_l_ua
        yy, xx = np.indices(garment_mask.shape)
        boundary_y = y1 + (xx - x1) * (y2 - y1) / (x2 - x1)
        lower_keep = yy >= boundary_y
        torso_zone = garment_arr.copy()
        torso_zone[:,:,3] = np.where(lower_keep, torso_zone[:,:,3], 0)
        warped_torso = cv2.warpAffine(torso_zone, torso_matrix, (aw, ah),
                                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        alpha = warped_torso[:,:,3:4].astype(np.float32)/255.0
        output = (output.astype(np.float32)*(1-alpha) + warped_torso.astype(np.float32)*alpha).astype(np.uint8)
        result["stage_reached"] = "full_render_complete"

        out_path = f"{OUTPUT_DIR}/generalization_{garment_label}.png"
        Image.fromarray(output).convert("RGB").save(out_path)
        result["output_path"] = out_path

    except Exception as e:
        result["error"] = f"mesh_or_render: {e}"
        return result

    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Building AvatarRepresentation (same golden avatar for all garments)...")
    rep = build_avatar_representation()
    aw, ah = rep.image.size

    print("\nDiscovering fresh garments (excluding all previously-used items)...")
    wardrobe = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    all_items = wardrobe.get("items", [])
    fresh_items = [
        item for item in all_items
        if item["item_id"] not in DEVELOPMENT_ITEM_IDS
        and item.get("category") == "top"
        and not item.get("is_archived", False)
    ]

    print(f"Found {len(fresh_items)} fresh garment(s).")

    test_garments = [{"item_id": GOLDEN_GARMENT_ID, "label": "golden_polo_control"}]
    for i, item in enumerate(fresh_items):
        test_garments.append({"item_id": item["item_id"], "label": f"fresh_{i+1}_{item.get('item_type','unknown')}"})

    print(f"\nRunning unchanged pipeline across {len(test_garments)} garment(s)...\n")

    results = []
    for g in test_garments:
        print(f"{'='*70}")
        print(f"{g['label']} ({g['item_id']})")
        print('='*70)
        r = run_pipeline_for_garment(g["item_id"], g["label"], rep, aw, ah)
        results.append(r)
        print(f"  stage_reached: {r['stage_reached']}")
        if r.get("error"):
            print(f"  ERROR: {r['error']}")
        else:
            print(f"  underarms: {r.get('underarms')}")
            print(f"  shoulders: {r.get('shoulders')}")
            print(f"  output: {r.get('output_path')}")
        print()

    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    for r in results:
        status = "✅ COMPLETE" if r["stage_reached"] == "full_render_complete" else f"❌ STOPPED AT: {r['stage_reached']}"
        print(f"  {r['label']:<35} {status}")
        if r.get("error"):
            print(f"      -> {r['error']}")


if __name__ == "__main__":
    main()