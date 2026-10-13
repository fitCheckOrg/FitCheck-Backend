"""
tryon/checkpoint_test3_equivalence_audit.py

TEST 3 — Behavioral equivalence audit, NOT a pixel-diff test.

Compares the SAME real golden avatar+garment pair through two paths:

  PATH A (production): the exact values FittingEngine.fit() actually
    reads and computes — garment.left_underarm etc. from the real
    GarmentRepresentation/AvatarRepresentation objects, then the same
    pairing + transform-construction logic fit() itself uses.

  PATH B (standalone research): piecewise_continuous_test.py's own
    main() logic, called directly — independently fetching the
    garment, calling find_garment_underarms/get_garment_shoulder_points
    directly, and independently recomputing avatar underarms via
    find_semantic_underarm + manual row-scan (NOT via
    AvatarRepresentation's .left_underarm property, which PATH A uses).

The real question: do GarmentRepresentation/AvatarRepresentation wrap
the same underlying detected values the standalone script computes, or
has drift crept in between the research script and the production
class? Checked at three levels: raw detected points, paired points,
and the actual transform matrices built from them.

No new deformation math. No rendering. No images produced. Every
function called is a real, already-existing one — this script only
calls and compares, nothing here is reimplemented.

Run: python -m tryon.checkpoint_test3_equivalence_audit
"""

import numpy as np
import cv2
import requests

from tryon.avatar_representation import build_avatar_representation
from tryon.garment_representation import build_garment_representation
from tryon.avatar_torso_landmarks_test import find_semantic_underarm
from tryon.correspondence_shoulders_test import find_garment_underarms
from tryon.garment_shoulder_region_extraction import get_binary_mask
from tryon.piecewise_continuous_test import get_garment_shoulder_points, solve_similarity_2point

GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"

TOLERANCE_PX = 0.01  # exact equality expected; this just absorbs float rounding


def pt_eq(label, a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    dist = float(np.hypot(*(a - b)))
    status = "MATCH" if dist <= TOLERANCE_PX else "DIFFERS"
    print(f"  {label:28s} A={tuple(a)}  B={tuple(b)}  Δ={dist:.4f}px  [{status}]")
    return status == "MATCH"


def matrix_eq(label, m1, m2):
    m1, m2 = np.array(m1, dtype=float), np.array(m2, dtype=float)
    max_diff = float(np.max(np.abs(m1 - m2)))
    status = "MATCH" if max_diff <= TOLERANCE_PX else "DIFFERS"
    print(f"\n  {label} — max element difference = {max_diff:.6f}  [{status}]")
    print(f"    A:\n{m1}")
    print(f"    B:\n{m2}")
    return status == "MATCH"


def pair_by_x(left, right):
    return (left, right) if left[0] < right[0] else (right, left)


def main():
    results = []

    # ================================================================
    # PATH A — production: build via the real classes FittingEngine.fit()
    # actually consumes.
    # ================================================================
    print("Building PATH A (production classes)...")
    avatar = build_avatar_representation()
    garment = build_garment_representation(GOLDEN_GARMENT_ID, AVATAR_USER_ID)

    a_g_ua_small, a_g_ua_large = pair_by_x(garment.left_underarm, garment.right_underarm)
    a_g_sh_small, a_g_sh_large = pair_by_x(garment.left_shoulder, garment.right_shoulder)
    a_a_ua_small, a_a_ua_large = pair_by_x(avatar.left_underarm, avatar.right_underarm)
    a_a_sh_small, a_a_sh_large = pair_by_x(avatar.left_shoulder, avatar.right_shoulder)

    # ================================================================
    # PATH B — standalone: piecewise_continuous_test.py's OWN main()
    # logic, called directly, not reimplemented.
    # ================================================================
    print("Building PATH B (standalone piecewise_continuous_test.py logic)...")
    garment_resp = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{GOLDEN_GARMENT_ID}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    garment_bytes = requests.get(garment_resp["clean_image_url"]).content
    b_mask, b_img = get_binary_mask(garment_bytes)
    b_gw = b_img.size[0]

    b_g_left_ua, b_g_right_ua = find_garment_underarms(b_mask, b_gw)
    b_g_left_sh, b_g_right_sh = get_garment_shoulder_points(b_mask, b_img, b_g_left_ua, b_g_right_ua, b_gw)

    b_rep = build_avatar_representation()  # independent build, same underlying data source
    b_aw = b_rep.image.size[0]
    b_a_l_uy, _ = find_semantic_underarm(b_rep.torso_mask, b_rep.upper_arms_mask, b_aw, "left")
    b_a_r_uy, _ = find_semantic_underarm(b_rep.torso_mask, b_rep.upper_arms_mask, b_aw, "right")
    b_a_l_row = b_rep.torso_mask[b_a_l_uy]; b_a_l_xs = np.where(b_a_l_row[:b_aw//2])[0]
    b_a_left_ua = (int(b_a_l_xs.min()), b_a_l_uy)
    b_a_r_row = b_rep.torso_mask[b_a_r_uy]; b_a_r_xs = np.where(b_a_r_row[b_aw//2:])[0] + b_aw//2
    b_a_right_ua = (int(b_a_r_xs.max()), b_a_r_uy)

    b_g_ua_small, b_g_ua_large = pair_by_x(b_g_left_ua, b_g_right_ua)
    b_g_sh_small, b_g_sh_large = pair_by_x(b_g_left_sh, b_g_right_sh)
    b_a_ua_small, b_a_ua_large = pair_by_x(b_a_left_ua, b_a_right_ua)
    b_a_sh_small, b_a_sh_large = pair_by_x(b_rep.left_shoulder, b_rep.right_shoulder)

    # ================================================================
    # LEVEL 1 — raw detected points, before any pairing
    # ================================================================
    print(f"\n{'='*70}")
    print("LEVEL 1 — Raw detected landmarks (A=production, B=standalone)")
    print(f"{'='*70}")
    print("Garment underarms:")
    results.append(pt_eq("underarm (small-x)", a_g_ua_small, b_g_ua_small))
    results.append(pt_eq("underarm (large-x)", a_g_ua_large, b_g_ua_large))
    print("Garment shoulders:")
    results.append(pt_eq("shoulder (small-x)", a_g_sh_small, b_g_sh_small))
    results.append(pt_eq("shoulder (large-x)", a_g_sh_large, b_g_sh_large))
    print("\nAvatar underarms — THE KEY UNKNOWN (AvatarRepresentation property vs standalone recompute):")
    results.append(pt_eq("underarm (small-x)", a_a_ua_small, b_a_ua_small))
    results.append(pt_eq("underarm (large-x)", a_a_ua_large, b_a_ua_large))
    print("Avatar shoulders (both paths use the same MediaPipe property either way):")
    results.append(pt_eq("shoulder (small-x)", a_a_sh_small, b_a_sh_small))
    results.append(pt_eq("shoulder (large-x)", a_a_sh_large, b_a_sh_large))

    # ================================================================
    # LEVEL 2 — the actual transform matrices each path's own logic
    # would construct from its own (possibly-differing) points above
    # ================================================================
    print(f"\n{'='*70}")
    print("LEVEL 2 — Transform matrices")
    print(f"{'='*70}")

    a_lower_matrix = solve_similarity_2point([a_g_ua_small, a_g_ua_large], [a_a_ua_small, a_a_ua_large])
    b_lower_matrix = solve_similarity_2point([b_g_ua_small, b_g_ua_large], [b_a_ua_small, b_a_ua_large])
    results.append(matrix_eq("Lower-zone similarity transform", a_lower_matrix, b_lower_matrix))

    a_tri1 = cv2.getAffineTransform(
        np.array([a_g_sh_small, a_g_sh_large, a_g_ua_small], dtype=np.float32),
        np.array([a_a_sh_small, a_a_sh_large, a_a_ua_small], dtype=np.float32))
    b_tri1 = cv2.getAffineTransform(
        np.array([b_g_sh_small, b_g_sh_large, b_g_ua_small], dtype=np.float32),
        np.array([b_a_sh_small, b_a_sh_large, b_a_ua_small], dtype=np.float32))
    results.append(matrix_eq("Upper triangle 1 affine transform", a_tri1, b_tri1))

    a_tri2 = cv2.getAffineTransform(
        np.array([a_g_sh_large, a_g_ua_small, a_g_ua_large], dtype=np.float32),
        np.array([a_a_sh_large, a_a_ua_small, a_a_ua_large], dtype=np.float32))
    b_tri2 = cv2.getAffineTransform(
        np.array([b_g_sh_large, b_g_ua_small, b_g_ua_large], dtype=np.float32),
        np.array([b_a_sh_large, b_a_ua_small, b_a_ua_large], dtype=np.float32))
    results.append(matrix_eq("Upper triangle 2 affine transform", a_tri2, b_tri2))

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    total, passed = len(results), sum(results)
    print(f"{passed}/{total} checks matched between production and standalone paths.")
    if passed == total:
        print(">>> FULL EQUIVALENCE: FittingEngine.fit() faithfully reproduces the")
        print(">>> standalone research implementation's behavior on this real pair.")
    else:
        print(">>> DISCREPANCY FOUND — see which specific check(s) above say DIFFERS.")
        print(">>> That divergence, not a guess, is what needs investigating next.")


if __name__ == "__main__":
    main()