"""
tryon/checkpoint4_experiment_E.py

CHECKPOINT 4 — Experiment E: local similarity-transform field, as
specified. NOT a production implementation — a geometric proof-of-
concept against the real golden pair, measuring whether pair-defined
transform blending solves the SPECIFIC insufficiency Checkpoint 3
localized (missing control geometry), not whether it reproduces the
old system's output.

HONEST SUBSTITUTION, flagged explicitly: real TopMesh (dense,
Poisson-sampled, poly2tri-triangulated) only exists in the Flutter/
Android pipeline — never generated in Python. This reuses the same
boundary-only ear-clipping mesh from the earlier rotation test.
E's blending formula itself only reads vertex coordinates, not mesh
connectivity, so this weakens ONLY the structural checks (inversion/
degenerate/overlap counts, which depend on triangle quality), not the
core transform-blending geometric behavior being tested.

E formula, exactly as specified:
  T_underarm = solve_similarity_2point(garment[ua_pair], avatar[ua_pair])
  T_shoulder = solve_similarity_2point(garment[sh_pair], avatar[sh_pair])
  for each point p:
    if p is (within epsilon of) an underarm landmark: w_ua=1, w_sh=0
    elif p is (within epsilon of) a shoulder landmark: w_ua=0, w_sh=1
    else:
      w_ua = 1 / (dist_to_segment(p, underarm_pair)^power + eps)
      w_sh = 1 / (dist_to_segment(p, shoulder_pair)^power + eps)
      normalize
    x'(p) = w_ua·T_underarm(p) + w_sh·T_shoulder(p)

Run: python -m tryon.checkpoint4_experiment_E
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageDraw

from tryon.avatar_representation import build_avatar_representation
from tryon.garment_representation import build_garment_representation
from tryon.piecewise_continuous_test import solve_similarity_2point

GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
OUTPUT_DIR = "outputs"

POWER = 2.0
EPSILON = 1.0
LANDMARK_OVERRIDE_RADIUS = 0.5  # px — within this of a defining landmark, force weight=1 to its own pair


# ---------------------------------------------------------------
# Mesh construction — SAME ear-clipping triangulator as the earlier
# rotation test (flagged substitution, see module docstring).
# ---------------------------------------------------------------

def trace_contour(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)


def simplify_contour(points, epsilon=2.0):
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    approx = cv2.approxPolyDP(pts, epsilon, closed=True)
    return approx.reshape(-1, 2)


def _point_in_triangle(p, a, b, c):
    def sign(p1, p2, p3):
        return (p1[0]-p3[0])*(p2[1]-p3[1]) - (p2[0]-p3[0])*(p1[1]-p3[1])
    d1, d2, d3 = sign(p, a, b), sign(p, b, c), sign(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def ear_clip_triangulate(polygon):
    area = sum(polygon[i][0]*polygon[(i+1) % len(polygon)][1] - polygon[(i+1) % len(polygon)][0]*polygon[i][1]
               for i in range(len(polygon)))
    if area < 0:
        polygon = polygon[::-1].copy()
    indices = list(range(len(polygon)))
    triangles = []
    guard = 0
    while len(indices) > 3 and guard < 20000:
        guard += 1
        ear_found = False
        n_idx = len(indices)
        for i in range(n_idx):
            i_prev, i_curr, i_next = indices[(i-1) % n_idx], indices[i], indices[(i+1) % n_idx]
            a, b, c = polygon[i_prev], polygon[i_curr], polygon[i_next]
            cross = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
            if cross <= 0:
                continue
            if any(_point_in_triangle(polygon[j], a, b, c) for j in indices if j not in (i_prev, i_curr, i_next)):
                continue
            triangles.append([i_prev, i_curr, i_next])
            indices.pop(i)
            ear_found = True
            break
        if not ear_found:
            break
    if len(indices) == 3:
        triangles.append(indices)
    return np.array(triangles)


def build_boundary_mesh(mask):
    raw = trace_contour(mask)
    boundary = simplify_contour(raw, epsilon=2.0)
    if np.allclose(boundary[0], boundary[-1]):
        boundary = boundary[:-1]
    triangles = ear_clip_triangulate(boundary)
    return boundary, triangles


# ---------------------------------------------------------------
# Verified Sutherland-Hodgman overlap detector — ported faithfully
# from tonight's independently ground-truth-verified implementation.
# ---------------------------------------------------------------

def ensure_ccw(poly):
    area = sum(poly[i][0]*poly[(i+1) % len(poly)][1] - poly[(i+1) % len(poly)][0]*poly[i][1] for i in range(len(poly)))
    return poly[::-1] if area < 0 else poly


def cross_sign(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])


def line_intersect(p1, p2, p3, p4):
    d1x, d1y = p2[0]-p1[0], p2[1]-p1[1]
    d2x, d2y = p4[0]-p3[0], p4[1]-p3[1]
    denom = d1x*d2y - d1y*d2x
    if abs(denom) < 1e-12:
        return None
    t = ((p3[0]-p1[0])*d2y - (p3[1]-p1[1])*d2x) / denom
    return (p1[0]+t*d1x, p1[1]+t*d1y)


def clip_polygon(subject, clip):
    output = subject
    for i in range(len(clip)):
        if not output:
            break
        a, b = clip[i], clip[(i+1) % len(clip)]
        inp, output = output, []
        for j in range(len(inp)):
            current, prev = inp[j], inp[(j-1) % len(inp)]
            cur_in, prev_in = cross_sign(a, b, current) >= 0, cross_sign(a, b, prev) >= 0
            if cur_in:
                if not prev_in:
                    ix = line_intersect(prev, current, a, b)
                    if ix: output.append(ix)
                output.append(current)
            elif prev_in:
                ix = line_intersect(prev, current, a, b)
                if ix: output.append(ix)
    return output


def polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    return abs(sum(poly[i][0]*poly[(i+1) % len(poly)][1] - poly[(i+1) % len(poly)][0]*poly[i][1] for i in range(len(poly)))) / 2.0


def detect_overlap(a1, b1, c1, a2, b2, c2):
    return polygon_area(clip_polygon(ensure_ccw([a1, b1, c1]), ensure_ccw([a2, b2, c2])))


# ---------------------------------------------------------------
# E's own geometry: segment distance, transform application, blending
# ---------------------------------------------------------------

def dist_to_segment(p, a, b):
    p, a, b = np.array(p, float), np.array(a, float), np.array(b, float)
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def apply_affine(matrix, pt):
    return (np.array(matrix) @ np.array([pt[0], pt[1], 1.0]))[:2]


def blend_at_point(p, ua_pair, sh_pair, T_ua, T_sh):
    """Returns (deformed_point, w_ua, w_sh) — the full E formula at one point."""
    for landmark in ua_pair:
        if np.linalg.norm(np.array(p) - np.array(landmark)) < LANDMARK_OVERRIDE_RADIUS:
            return apply_affine(T_ua, p), 1.0, 0.0
    for landmark in sh_pair:
        if np.linalg.norm(np.array(p) - np.array(landmark)) < LANDMARK_OVERRIDE_RADIUS:
            return apply_affine(T_sh, p), 0.0, 1.0

    d_ua = dist_to_segment(p, *ua_pair)
    d_sh = dist_to_segment(p, *sh_pair)
    w_ua_raw = 1.0 / (d_ua ** POWER + EPSILON)
    w_sh_raw = 1.0 / (d_sh ** POWER + EPSILON)
    total = w_ua_raw + w_sh_raw
    w_ua, w_sh = w_ua_raw / total, w_sh_raw / total

    pred_ua = apply_affine(T_ua, p)
    pred_sh = apply_affine(T_sh, p)
    deformed = w_ua * pred_ua + w_sh * pred_sh
    return deformed, w_ua, w_sh


def pair_by_x(left, right):
    return (left, right) if left[0] < right[0] else (right, left)


def render_mesh(vertices_rest, vertices_deformed, triangles, texture_arr, canvas_w, canvas_h):
    out = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    tex_h, tex_w = texture_arr.shape[:2]
    for tri in triangles:
        rest, dst = vertices_rest[tri], vertices_deformed[tri]
        x0, y0 = np.floor(dst.min(axis=0)).astype(int)
        x1, y1 = np.ceil(dst.max(axis=0)).astype(int)
        x0, y0, x1, y1 = max(x0,0), max(y0,0), min(x1,canvas_w), min(y1,canvas_h)
        if x1 <= x0 or y1 <= y0:
            continue
        denom = (dst[1,1]-dst[2,1])*(dst[0,0]-dst[2,0]) + (dst[2,0]-dst[1,0])*(dst[0,1]-dst[2,1])
        if abs(denom) < 1e-9:
            continue
        for py in range(y0, y1):
            for px in range(x0, x1):
                p = np.array([px+0.5, py+0.5])
                u = ((dst[1,1]-dst[2,1])*(p[0]-dst[2,0]) + (dst[2,0]-dst[1,0])*(p[1]-dst[2,1])) / denom
                v = ((dst[2,1]-dst[0,1])*(p[0]-dst[2,0]) + (dst[0,0]-dst[2,0])*(p[1]-dst[2,1])) / denom
                w = 1 - u - v
                if u < 0 or v < 0 or w < 0:
                    continue
                src = u*rest[0] + v*rest[1] + w*rest[2]
                sx, sy = int(np.clip(src[0],0,tex_w-1)), int(np.clip(src[1],0,tex_h-1))
                out[py, px] = texture_arr[sy, sx]
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Building real AvatarRepresentation and GarmentRepresentation...")
    avatar = build_avatar_representation()
    garment = build_garment_representation(GOLDEN_GARMENT_ID, AVATAR_USER_ID)
    if not garment.shoulders_available:
        raise RuntimeError("E requires both pairs; this golden garment must have shoulders_available=True "
                            "(confirmed True in Checkpoint 0 — if this fires, something upstream changed).")

    g_ua = pair_by_x(garment.left_underarm, garment.right_underarm)
    g_sh = pair_by_x(garment.left_shoulder, garment.right_shoulder)
    a_ua = pair_by_x(avatar.left_underarm, avatar.right_underarm)
    a_sh = pair_by_x(avatar.left_shoulder, avatar.right_shoulder)

    T_ua = solve_similarity_2point(list(g_ua), list(a_ua))
    T_sh = solve_similarity_2point(list(g_sh), list(a_sh))
    print(f"\nT_underarm matrix:\n{np.array(T_ua)}")
    print(f"T_shoulder matrix:\n{np.array(T_sh)}")

    print("\nBuilding boundary-only mesh (flagged substitution — see module docstring)...")
    vertices, triangles = build_boundary_mesh(garment.mask)
    print(f"Mesh: {len(vertices)} vertices, {len(triangles)} triangles")

    deformed = np.array([blend_at_point(v, g_ua, g_sh, T_ua, T_sh)[0] for v in vertices])

    # ============================================================
    # 1-2. Landmark positional error
    # ============================================================
    print(f"\n{'='*70}\n1-2. LANDMARK POSITIONAL ERROR\n{'='*70}")
    for name, pt, target in [("underarm_small", g_ua[0], a_ua[0]), ("underarm_large", g_ua[1], a_ua[1]),
                              ("shoulder_small", g_sh[0], a_sh[0]), ("shoulder_large", g_sh[1], a_sh[1])]:
        d, w_ua, w_sh = blend_at_point(pt, g_ua, g_sh, T_ua, T_sh)
        err = float(np.hypot(*(d - np.array(target))))
        print(f"  {name:16s} error={err:.4f}px  (w_ua={w_ua:.3f}, w_sh={w_sh:.3f})")

    # ============================================================
    # 3. Shared-field continuity, including AT the override boundary
    # ============================================================
    print(f"\n{'='*70}\n3. SHARED-FIELD CONTINUITY\n{'='*70}")
    mid_ua = np.mean(g_ua, axis=0)
    mid_sh = np.mean(g_sh, axis=0)
    path = [mid_ua + t*(mid_sh - mid_ua) for t in np.linspace(0, 1, 11)]
    prev = None
    max_step_jump = 0.0
    for i, p in enumerate(path):
        d, w_ua, w_sh = blend_at_point(p, g_ua, g_sh, T_ua, T_sh)
        step = float(np.hypot(*(d - prev))) if prev is not None else 0.0
        max_step_jump = max(max_step_jump, step) if i > 1 else max_step_jump
        print(f"  t={i/10:.1f}  w_ua={w_ua:.3f} w_sh={w_sh:.3f}  deformed={tuple(d)}  step_from_prev={step:.2f}px")
        prev = d
    print(f"  max inter-step jump (excluding first step) = {max_step_jump:.2f}px — large relative jump would indicate non-smooth blending")

    print("\n  Override-boundary check (does the hard override create a visible seam?):")
    for name, landmark in [("underarm_small", g_ua[0]), ("shoulder_small", g_sh[0])]:
        at_exact, _, _ = blend_at_point(landmark, g_ua, g_sh, T_ua, T_sh)
        just_outside_pt = np.array(landmark) + np.array([LANDMARK_OVERRIDE_RADIUS*1.5, 0])
        at_outside, w_ua_o, w_sh_o = blend_at_point(just_outside_pt, g_ua, g_sh, T_ua, T_sh)
        jump = float(np.hypot(*(at_exact - at_outside)))
        print(f"    {name}: at landmark={tuple(at_exact)}  just outside override radius={tuple(at_outside)}"
              f"  (w_ua={w_ua_o:.4f}) jump={jump:.3f}px")

    # ============================================================
    # 4-6. Mesh structural integrity
    # ============================================================
    print(f"\n{'='*70}\n4-6. MESH STRUCTURAL INTEGRITY\n{'='*70}")
    ref_winding = None
    inverted, degenerate = 0, 0
    for tri in triangles:
        a, b, c = deformed[tri]
        area2 = (b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])
        if abs(area2) < 1e-6:
            degenerate += 1
            continue
        sign = 1 if area2 > 0 else -1
        if ref_winding is None:
            ref_winding = sign
        elif sign != ref_winding:
            inverted += 1
    print(f"  Inverted triangles: {inverted} / {len(triangles)}")
    print(f"  Degenerate triangles: {degenerate} / {len(triangles)}")

    overlap_pairs, max_overlap = 0, 0.0
    for i in range(len(triangles)):
        ti = triangles[i]
        for j in range(i+1, len(triangles)):
            tj = triangles[j]
            if set(ti.tolist()) & set(tj.tolist()):
                continue
            area = detect_overlap(*deformed[ti], *deformed[tj])
            if area > 0.5:
                overlap_pairs += 1
                max_overlap = max(max_overlap, area)
    print(f"  Overlap pairs: {overlap_pairs}  (max area={max_overlap:.2f}px²)")

    # ============================================================
    # 7-8. Hem and sleeve behavior
    # ============================================================
    print(f"\n{'='*70}\n7-8. HEM AND SLEEVE BEHAVIOR\n{'='*70}")
    mask = garment.mask
    ys, xs = np.where(mask)
    bottom_y = ys.max()
    bottom_xs = xs[ys == bottom_y]
    test_points = {
        "left_hem": (float(bottom_xs.min()), float(bottom_y)),
        "center_hem": (float(np.median(bottom_xs)), float(bottom_y)),
        "right_hem": (float(bottom_xs.max()), float(bottom_y)),
        "left_sleeve_end": (float(xs.min()), float(ys[xs == xs.min()].mean())),
        "right_sleeve_end": (float(xs.max()), float(ys[xs == xs.max()].mean())),
    }
    predicted = {}
    for name, pt in test_points.items():
        d, w_ua, w_sh = blend_at_point(pt, g_ua, g_sh, T_ua, T_sh)
        predicted[name] = d
        print(f"  {name:18s} garment={pt}  →  E_deformed={tuple(d)}  (w_ua={w_ua:.3f}, w_sh={w_sh:.3f})")

    # ============================================================
    # 9. Rotation behavior — local effective rotation at sample points,
    #    via a small reference vector transformed through the SAME
    #    blend (recomputing weights at the offset point too).
    # ============================================================
    print(f"\n{'='*70}\n9. ROTATION BEHAVIOR (local effective rotation at sample points)\n{'='*70}")
    def local_rotation_deg(p, eps=2.0):
        p = np.array(p, float)
        d0, _, _ = blend_at_point(p, g_ua, g_sh, T_ua, T_sh)
        d1, _, _ = blend_at_point(p + [eps, 0], g_ua, g_sh, T_ua, T_sh)
        v = d1 - d0
        return float(np.degrees(np.arctan2(v[1], v[0])))
    for name, pt in [("near_underarm", mid_ua), ("midpoint", np.mean([mid_ua, mid_sh], axis=0)),
                      ("near_shoulder", mid_sh), ("left_hem", test_points["left_hem"])]:
        print(f"  {name:16s} local effective rotation = {local_rotation_deg(pt):.2f}°  "
              f"(reference: T_underarm alone = flat 0° baseline direction)")

    # ============================================================
    # 10. Render for direct visual comparison against Checkpoint 0
    # ============================================================
    garment_arr = np.array(garment.image.convert("RGBA"))
    avatar_rgba = np.array(avatar.image.convert("RGBA"))
    aw, ah = avatar.image.size
    layer = render_mesh(vertices, deformed, triangles, garment_arr, aw, ah)
    composite = avatar_rgba.copy()
    alpha = layer[:, :, 3:4].astype(float) / 255.0
    composite[:, :, :3] = (avatar_rgba[:, :, :3]*(1-alpha) + layer[:, :, :3]*alpha).astype(np.uint8)
    Image.fromarray(composite).convert("RGB").save(f"{OUTPUT_DIR}/checkpoint4_E_render.png")

    annotated = Image.fromarray(composite).convert("RGB")
    draw = ImageDraw.Draw(annotated)
    colors = {"left_hem":(255,0,0),"center_hem":(255,128,0),"right_hem":(255,0,255),
              "left_sleeve_end":(0,255,255),"right_sleeve_end":(0,255,0)}
    for name, pt in predicted.items():
        x, y = pt
        draw.ellipse([x-6,y-6,x+6,y+6], outline=colors[name], width=3)
    for pt in [*a_ua, *a_sh]:
        x, y = pt
        draw.ellipse([x-5,y-5,x+5,y+5], outline=(255,255,0), width=2)
    annotated.save(f"{OUTPUT_DIR}/checkpoint4_E_render_annotated.png")

    print(f"\nSaved: {OUTPUT_DIR}/checkpoint4_E_render.png")
    print(f"Saved: {OUTPUT_DIR}/checkpoint4_E_render_annotated.png")
    print(f"\n>>> Compare directly against outputs/checkpoint0_render_annotated.png —")
    print(f">>> same garment, same avatar, same 5 tracked points, same yellow-anchor convention.")


if __name__ == "__main__":
    main()