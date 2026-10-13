"""
tryon/checkpoint0_baseline_validation.py

CHECKPOINT 0 / TEST 1 / TEST 2 — validates the EXISTING, already-
shipped production pipeline (FittingEngine.fit() + Compositor.composite(),
called completely unmodified) against the real golden avatar+garment
pair.

Introduces ZERO new deformation math, mesh representation, or
rendering code. Every transform and every composited pixel in the
saved image comes directly from the real production functions,
called exactly as tryon_pipeline.py already calls them.

The ONLY thing this script adds is instrumentation: numeric
diagnostics computed by independently re-deriving the same real
landmark positions (via the same real detection functions) and
re-applying the SAME real transform matrices FittingEngine.fit()
already computed internally — not different math, just making
internal values externally visible for inspection.

Run: python -m tryon.checkpoint0_baseline_validation
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageDraw

from tryon.avatar_representation import build_avatar_representation
from tryon.garment_representation import build_garment_representation
from tryon.fitting_engine import FittingEngine
from tryon.compositor import Compositor
from tryon.piecewise_continuous_test import solve_similarity_2point

GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
OUTPUT_DIR = "outputs"


def decompose_similarity(matrix):
    """Pure instrumentation — reads scale/rotation/translation back out
    of a similarity matrix solve_similarity_2point already returns.
    Does not change how that matrix was computed."""
    a_real, a_imag = matrix[0, 0], matrix[1, 0]
    scale = float(np.hypot(a_real, a_imag))
    rotation_deg = float(np.degrees(np.arctan2(a_imag, a_real)))
    translation = (float(matrix[0, 2]), float(matrix[1, 2]))
    return scale, rotation_deg, translation


def apply_affine(matrix, pt):
    return (matrix @ np.array([pt[0], pt[1], 1.0]))[:2]


def point_in_triangle(p, a, b, c):
    def sign(p1, p2, p3):
        return (p1[0]-p3[0])*(p2[1]-p3[1]) - (p2[0]-p3[0])*(p1[1]-p3[1])
    d1, d2, d3 = sign(p, a, b), sign(p, b, c), sign(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ============================================================
    # REAL PRODUCTION PIPELINE — exactly as tryon_pipeline.py calls it.
    # ============================================================
    print("Building REAL AvatarRepresentation and GarmentRepresentation...")
    avatar = build_avatar_representation()
    garment = build_garment_representation(GOLDEN_GARMENT_ID, AVATAR_USER_ID)

    print(f"\nGarment asset check:")
    print(f"  image size: {garment.width} x {garment.height}")
    print(f"  shoulders_available: {garment.shoulders_available}")
    print(f"  underarms: L={garment.left_underarm}  R={garment.right_underarm}")
    if garment.shoulders_available:
        print(f"  shoulders: L={garment.left_shoulder}  R={garment.right_shoulder}")

    print(f"\nAvatar landmark check:")
    print(f"  underarms: L={avatar.left_underarm}  R={avatar.right_underarm}")
    print(f"  shoulders (MediaPipe): L={avatar.left_shoulder}  R={avatar.right_shoulder}")

    print("\nCalling REAL FittingEngine.fit() — unmodified...")
    fitting_result = FittingEngine.fit(garment, avatar)

    print("Calling REAL Compositor.composite() — unmodified...")
    rendered = Compositor.composite(avatar.image, fitting_result)
    rendered.convert("RGB").save(f"{OUTPUT_DIR}/checkpoint0_render.png")
    print(f"Saved: {OUTPUT_DIR}/checkpoint0_render.png")

    # ============================================================
    # DIAGNOSTICS — re-deriving the SAME real values FittingEngine.fit()
    # already computed internally, purely for external visibility.
    # ============================================================
    g_s_ua = garment.left_underarm if garment.left_underarm[0] < garment.right_underarm[0] else garment.right_underarm
    g_l_ua = garment.right_underarm if garment.left_underarm[0] < garment.right_underarm[0] else garment.left_underarm
    a_s_ua = avatar.left_underarm if avatar.left_underarm[0] < avatar.right_underarm[0] else avatar.right_underarm
    a_l_ua = avatar.right_underarm if avatar.left_underarm[0] < avatar.right_underarm[0] else avatar.left_underarm

    lower_matrix = solve_similarity_2point([g_s_ua, g_l_ua], [a_s_ua, a_l_ua])
    scale, rotation_deg, translation = decompose_similarity(lower_matrix)

    print(f"\n{'='*60}")
    print("TEST 1 — UNDERARM-ONLY BASELINE (lower-zone transform)")
    print(f"{'='*60}")
    print(f"scale = {scale:.4f}")
    print(f"rotation = {rotation_deg:.2f}°")
    print(f"translation = {translation}")

    pred_s_ua = apply_affine(lower_matrix, g_s_ua)
    pred_l_ua = apply_affine(lower_matrix, g_l_ua)
    err_s = np.hypot(*(pred_s_ua - np.array(a_s_ua)))
    err_l = np.hypot(*(pred_l_ua - np.array(a_l_ua)))
    landmark_max_error = max(err_s, err_l)
    print(f"underarm landmark errors: small-x={err_s:.4f}px  large-x={err_l:.4f}px")
    print(f"landmark_max_error = {landmark_max_error:.4f}px")
    print(f"render dimensions = {rendered.size}")

    if garment.shoulders_available:
        g_s_sh = garment.left_shoulder if garment.left_shoulder[0] < garment.right_shoulder[0] else garment.right_shoulder
        g_l_sh = garment.right_shoulder if garment.left_shoulder[0] < garment.right_shoulder[0] else garment.left_shoulder
        a_s_sh = avatar.left_shoulder if avatar.left_shoulder[0] < avatar.right_shoulder[0] else avatar.right_shoulder
        a_l_sh = avatar.right_shoulder if avatar.left_shoulder[0] < avatar.right_shoulder[0] else avatar.left_shoulder

        tri1_src = [g_s_sh, g_l_sh, g_s_ua]
        tri1_dst = [a_s_sh, a_l_sh, a_s_ua]
        tri2_src = [g_l_sh, g_s_ua, g_l_ua]
        tri2_dst = [a_l_sh, a_s_ua, a_l_ua]

        tri1_matrix = cv2.getAffineTransform(np.array(tri1_src, dtype=np.float32), np.array(tri1_dst, dtype=np.float32))
        tri2_matrix = cv2.getAffineTransform(np.array(tri2_src, dtype=np.float32), np.array(tri2_dst, dtype=np.float32))

        pred_s_sh = apply_affine(tri1_matrix, g_s_sh)
        pred_l_sh = apply_affine(tri2_matrix, g_l_sh)
        sh_err_s = np.hypot(*(pred_s_sh - np.array(a_s_sh)))
        sh_err_l = np.hypot(*(pred_l_sh - np.array(a_l_sh)))
        print(f"\nshoulder landmark errors: small-x={sh_err_s:.4f}px  large-x={sh_err_l:.4f}px")
        print(f"(upper zone: two independent triangle affine fits, per real FittingEngine.fit())")

    # ============================================================
    # TEST 2 — 5 real UNCONSTRAINED garment points, tracked through
    # whichever real transform governs their zone.
    # ============================================================
    print(f"\n{'='*60}")
    print("TEST 2 — UNCONSTRAINED GARMENT POINTS")
    print(f"{'='*60}")

    mask = garment.mask
    ys, xs = np.where(mask)
    bottom_y = ys.max()
    bottom_row_xs = xs[ys == bottom_y]
    left_hem = (float(bottom_row_xs.min()), float(bottom_y))
    right_hem = (float(bottom_row_xs.max()), float(bottom_y))
    center_hem = (float(np.median(bottom_row_xs)), float(bottom_y))
    left_sleeve = (float(xs.min()), float(ys[xs == xs.min()].mean()))
    right_sleeve = (float(xs.max()), float(ys[xs == xs.max()].mean()))

    test_points = {
        "left_hem": left_hem, "center_hem": center_hem, "right_hem": right_hem,
        "left_sleeve_end": left_sleeve, "right_sleeve_end": right_sleeve,
    }

    x1, y1 = g_s_ua
    x2, y2 = g_l_ua
    predicted = {}
    for name, pt in test_points.items():
        boundary_y_at_x = y1 + (pt[0]-x1)*(y2-y1)/(x2-x1) if x2 != x1 else y1
        if pt[1] >= boundary_y_at_x:
            pred = apply_affine(lower_matrix, pt)
            zone = "lower(similarity)"
        elif garment.shoulders_available and point_in_triangle(pt, *tri1_src):
            pred = apply_affine(tri1_matrix, pt)
            zone = "upper-tri1"
        elif garment.shoulders_available and point_in_triangle(pt, *tri2_src):
            pred = apply_affine(tri2_matrix, pt)
            zone = "upper-tri2"
        else:
            pred = apply_affine(lower_matrix, pt)
            zone = "lower(fallback, outside both triangles)"
        predicted[name] = pred
        print(f"  {name:18s}  garment={pt}  →  predicted_avatar_space={tuple(pred)}  [{zone}]")

    draw = ImageDraw.Draw(rendered)
    colors = {"left_hem": (255,0,0), "center_hem": (255,128,0), "right_hem": (255,0,255),
              "left_sleeve_end": (0,255,255), "right_sleeve_end": (0,255,0)}
    for name, pred in predicted.items():
        x, y = pred
        draw.ellipse([x-6, y-6, x+6, y+6], outline=colors[name], width=3)
    for pt, color in [(a_s_ua,(255,255,0)), (a_l_ua,(255,255,0))]:
        x, y = pt
        draw.ellipse([x-5, y-5, x+5, y+5], outline=color, width=2)
    rendered.convert("RGB").save(f"{OUTPUT_DIR}/checkpoint0_render_annotated.png")
    print(f"\nSaved annotated: {OUTPUT_DIR}/checkpoint0_render_annotated.png")
    print(f"  (colored circles = tracked unconstrained points; yellow = underarm controls)")
    print(f"\n>>> VISUAL INSPECTION NEEDED: do the 5 tracked points land somewhere that")
    print(f">>> actually looks like a hem and sleeve ends on the rendered avatar, or")
    print(f">>> somewhere clearly wrong?")


if __name__ == "__main__":
    main()