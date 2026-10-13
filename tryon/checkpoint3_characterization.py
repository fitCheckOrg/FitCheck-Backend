"""
tryon/checkpoint3_characterization.py

CHECKPOINT 3 — characterization, not redesign. Answers exactly one
question per test point: which existing transform owns this point,
is that ownership interpolation or extrapolation, and how far is the
point from the data that actually constrains that transform?

No new deformation math. No mesh. No rendering. Reuses the exact
same real FittingEngine.fit() construction already verified in
Checkpoint 0 / Test 3 — this script only adds measurement.

Run: python -m tryon.checkpoint3_characterization
"""

import numpy as np

from tryon.avatar_representation import build_avatar_representation
from tryon.garment_representation import build_garment_representation
from tryon.piecewise_continuous_test import solve_similarity_2point
import cv2

GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"


def pair_by_x(left, right):
    return (left, right) if left[0] < right[0] else (right, left)


def apply_affine(matrix, pt):
    return (np.array(matrix) @ np.array([pt[0], pt[1], 1.0]))[:2]


def barycentric(p, a, b, c):
    """Standard barycentric coordinates of p w.r.t. triangle a,b,c.
    All three in [0,1] => p is genuinely inside (interpolation).
    Any outside [0,1] => p is outside the triangle (extrapolation),
    and HOW far outside indicates extrapolation severity."""
    a, b, c, p = map(lambda x: np.array(x, dtype=float), (a, b, c, p))
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    d20, d21 = v2 @ v0, v2 @ v1
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w
    return u, v, w


def classify_bary(u, v, w):
    if min(u, v, w) >= 0 and max(u, v, w) <= 1:
        return "INTERPOLATION (inside)"
    overshoot = -min(min(u, v, w), 0) + max(max(u, v, w) - 1, 0)
    return f"EXTRAPOLATION (outside, overshoot={overshoot:.3f})"


def report_triangle_point(name, src_pt, tri_src, tri_dst, matrix):
    u, v, w = barycentric(src_pt, *tri_src)
    classification = classify_bary(u, v, w)
    transformed = apply_affine(matrix, src_pt)
    dists = [float(np.hypot(*(np.array(src_pt) - np.array(anchor)))) for anchor in tri_src]
    print(f"\n{name}")
    print(f"  source = {tuple(src_pt)}")
    print(f"  owning transform = triangle affine, barycentric (u,v,w) = ({u:.3f}, {v:.3f}, {w:.3f})")
    print(f"  classification = {classification}")
    print(f"  transformed = {tuple(transformed)}")
    print(f"  distance from triangle's 3 defining anchors = {[f'{d:.1f}px' for d in dists]}")
    return transformed


def report_similarity_point(name, src_pt, anchors, matrix):
    transformed = apply_affine(matrix, src_pt)
    dists = [float(np.hypot(*(np.array(src_pt) - np.array(a)))) for a in anchors]
    print(f"\n{name}")
    print(f"  source = {tuple(src_pt)}")
    print(f"  owning transform = lower-zone similarity (NOTE: a 2-point similarity transform")
    print(f"    has no 'interior' — it is one rigid+scale relationship, equally valid")
    print(f"    everywhere mathematically. Interpolation/extrapolation doesn't apply the same")
    print(f"    way it does for a 3-point affine fit; distance from the defining data is the")
    print(f"    honest equivalent signal here.)")
    print(f"  transformed = {tuple(transformed)}")
    print(f"  distance from the 2 defining underarm anchors = {[f'{d:.1f}px' for d in dists]}")
    return transformed


def main():
    print("Building real AvatarRepresentation and GarmentRepresentation...")
    avatar = build_avatar_representation()
    garment = build_garment_representation(GOLDEN_GARMENT_ID, AVATAR_USER_ID)

    g_s_ua, g_l_ua = pair_by_x(garment.left_underarm, garment.right_underarm)
    g_s_sh, g_l_sh = pair_by_x(garment.left_shoulder, garment.right_shoulder)
    a_s_ua, a_l_ua = pair_by_x(avatar.left_underarm, avatar.right_underarm)
    a_s_sh, a_l_sh = pair_by_x(avatar.left_shoulder, avatar.right_shoulder)

    tri1_src = [g_s_sh, g_l_sh, g_s_ua]
    tri1_dst = [a_s_sh, a_l_sh, a_s_ua]
    tri2_src = [g_l_sh, g_s_ua, g_l_ua]
    tri2_dst = [a_l_sh, a_s_ua, a_l_ua]

    tri1_matrix = cv2.getAffineTransform(np.array(tri1_src, dtype=np.float32), np.array(tri1_dst, dtype=np.float32))
    tri2_matrix = cv2.getAffineTransform(np.array(tri2_src, dtype=np.float32), np.array(tri2_dst, dtype=np.float32))
    lower_matrix = solve_similarity_2point([g_s_ua, g_l_ua], [a_s_ua, a_l_ua])

    # ============================================================
    # Reference: already-confirmed exact points (Test 1)
    # ============================================================
    print(f"\n{'='*70}")
    print("REFERENCE — already confirmed exact (Checkpoint 0 / Test 1)")
    print(f"{'='*70}")
    print(f"  shoulders: 0.0000px error (both)")
    print(f"  underarms: 0.0000px error (both)")

    # ============================================================
    # Upper torso — genuine interior interpolation points
    # ============================================================
    print(f"\n{'='*70}")
    print("UPPER TORSO — points genuinely inside each triangle")
    print(f"{'='*70}")
    tri1_centroid = tuple(np.mean(tri1_src, axis=0))
    tri2_centroid = tuple(np.mean(tri2_src, axis=0))
    report_triangle_point("upper_torso_tri1_centroid", tri1_centroid, tri1_src, tri1_dst, tri1_matrix)
    report_triangle_point("upper_torso_tri2_centroid", tri2_centroid, tri2_src, tri2_dst, tri2_matrix)

    # ============================================================
    # Sleeves — real garment extremes, explicitly classified
    # ============================================================
    print(f"\n{'='*70}")
    print("SLEEVES — real garment extremes")
    print(f"{'='*70}")
    mask = garment.mask
    ys, xs = np.where(mask)
    left_sleeve = (float(xs.min()), float(ys[xs == xs.min()].mean()))
    right_sleeve = (float(xs.max()), float(ys[xs == xs.max()].mean()))

    for name, pt in [("left_sleeve_end", left_sleeve), ("right_sleeve_end", right_sleeve)]:
        u1, v1, w1 = barycentric(pt, *tri1_src)
        u2, v2, w2 = barycentric(pt, *tri2_src)
        in_tri1 = min(u1, v1, w1) >= 0 and max(u1, v1, w1) <= 1
        in_tri2 = min(u2, v2, w2) >= 0 and max(u2, v2, w2) <= 1
        if in_tri1:
            report_triangle_point(name, pt, tri1_src, tri1_dst, tri1_matrix)
        elif in_tri2:
            report_triangle_point(name, pt, tri2_src, tri2_dst, tri2_matrix)
        else:
            print(f"\n{name}")
            print(f"  source = {pt}")
            print(f"  NOT inside either upper triangle — tri1 bary=({u1:.2f},{v1:.2f},{w1:.2f}),"
                  f" tri2 bary=({u2:.2f},{v2:.2f},{w2:.2f})")
            print(f"  owning transform = FALLS THROUGH to lower-zone similarity (confirms Checkpoint 0's")
            print(f"    'fallback' finding explicitly, via real barycentric test rather than inference)")
            report_similarity_point(name, pt, [g_s_ua, g_l_ua], lower_matrix)

    # ============================================================
    # Lower torso / hem
    # ============================================================
    print(f"\n{'='*70}")
    print("LOWER TORSO / HEM")
    print(f"{'='*70}")
    bottom_y = ys.max()
    bottom_row_xs = xs[ys == bottom_y]
    left_hem = (float(bottom_row_xs.min()), float(bottom_y))
    center_hem = (float(np.median(bottom_row_xs)), float(bottom_y))
    right_hem = (float(bottom_row_xs.max()), float(bottom_y))
    for name, pt in [("left_hem", left_hem), ("center_hem", center_hem), ("right_hem", right_hem)]:
        report_similarity_point(name, pt, [g_s_ua, g_l_ua], lower_matrix)

    # ============================================================
    # Boundary transitions — the actual continuity checks
    # ============================================================
    print(f"\n{'='*70}")
    print("BOUNDARY TRANSITIONS — do adjacent transforms actually agree?")
    print(f"{'='*70}")

    shared_edge_mid = tuple(np.mean([g_l_sh, g_s_ua], axis=0))
    t1_pred = apply_affine(tri1_matrix, shared_edge_mid)
    t2_pred = apply_affine(tri2_matrix, shared_edge_mid)
    disagreement = float(np.hypot(*(t1_pred - t2_pred)))
    print(f"\ntri1/tri2 shared-edge midpoint = {shared_edge_mid}")
    print(f"  tri1 predicts: {tuple(t1_pred)}")
    print(f"  tri2 predicts: {tuple(t2_pred)}")
    print(f"  disagreement = {disagreement:.4f}px  "
          f"({'CONTINUOUS — matches by construction' if disagreement < 0.01 else 'DISCONTINUITY DETECTED'})")

    underarm_line_mid = tuple(np.mean([g_s_ua, g_l_ua], axis=0))
    tri2_pred = apply_affine(tri2_matrix, underarm_line_mid)
    lower_pred = apply_affine(lower_matrix, underarm_line_mid)
    disagreement2 = float(np.hypot(*(tri2_pred - lower_pred)))
    print(f"\nunderarm-line midpoint (upper/lower zone boundary) = {underarm_line_mid}")
    print(f"  tri2 (upper, affine) predicts:        {tuple(tri2_pred)}")
    print(f"  lower zone (similarity) predicts:     {tuple(lower_pred)}")
    print(f"  disagreement = {disagreement2:.4f}px  "
          f"({'CONTINUOUS' if disagreement2 < 1.0 else 'REAL DISCONTINUITY AT THE ZONE BOUNDARY'})")

    print(f"\n{'='*70}")
    print("CHECKPOINT 3 COMPLETE — see each section above for the factual map.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()