"""
tryon/idw_rotation_test.py

Targeted V1 test: does a real avatar+garment correspondence have a
meaningful orientation difference, and if so, does mesh-based IDW
deformation reproduce it as well as the proven backend transform does?

Compares two independently-computed outcomes for the SAME real
correspondence (underarms only, matching solve_similarity_2point's
own 2-point case exactly):

  A. REFERENCE — the actual proven backend method: solve_similarity_2point
     + cv2.warpAffine on the whole garment image. Unmodified, exactly as
     it runs in production today.

  B. IDW — mesh deformed with IDW using the SAME two underarm points as
     its only controls.

     NOTE ON MESH CONSTRUCTION: this uses a BOUNDARY-ONLY mesh (ear-
     clipping triangulation on the traced+simplified silhouette, no
     Poisson interior points), not the full dense TopMesh pipeline.
     Swapped in specifically to avoid requiring a C++ compiler on
     Windows for the `triangle` package (see prior failed install —
     no prebuilt wheel exists for it, source build needs MSVC Build
     Tools, a heavy detour for one targeted test). A boundary-only
     mesh was already established earlier in this project as clean
     and overlap-free at rest; it's sufficient for testing orientation
     behavior under IDW, which this test is specifically about.

Both underarms land exactly on target under EITHER method — that's
guaranteed by construction and proves nothing. The real signal is
where the garment's own SHOULDER points end up under each method,
since neither uses them as a control point. Real pixel distance
between the two outcomes, for those points specifically, is the
actual measurement.

No external dependencies beyond what's already installed (numpy,
opencv-python, Pillow, requests).

Run: python -m tryon.idw_rotation_test
"""

import os
import numpy as np
import cv2
from PIL import Image

from tryon.avatar_representation import build_avatar_representation
from tryon.avatar_torso_landmarks_test import find_semantic_underarm
from tryon.correspondence_shoulders_test import find_garment_underarms
from tryon.garment_shoulder_region_extraction import get_binary_mask
from tryon.piecewise_continuous_test import solve_similarity_2point, get_garment_shoulder_points

GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
OUTPUT_DIR = "outputs"


# ---------------------------------------------------------------
# Contour tracing + simplification — same methods validated
# throughout tonight's TopMesh work.
# ---------------------------------------------------------------

def trace_contour(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)


def simplify_contour(points, epsilon=2.0):
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    approx = cv2.approxPolyDP(pts, epsilon, closed=True)
    return approx.reshape(-1, 2)


# ---------------------------------------------------------------
# Ear-clipping triangulation — pure Python/numpy, no compiled
# dependency. Standard algorithm: repeatedly clip a convex vertex
# whose triangle contains no other polygon vertex, until 3 remain.
# Fine for a simplified contour's typical point count (tens of
# points); this is a one-off research script, not a hot path.
# ---------------------------------------------------------------

def _point_in_triangle(p, a, b, c):
    def sign(p1, p2, p3):
        return (p1[0]-p3[0])*(p2[1]-p3[1]) - (p2[0]-p3[0])*(p1[1]-p3[1])
    d1, d2, d3 = sign(p, a, b), sign(p, b, c), sign(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def ear_clip_triangulate(polygon):
    # Ensure CCW winding (required for the convexity test below).
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
            if cross <= 0:  # reflex vertex, not a valid ear
                continue
            if any(_point_in_triangle(polygon[j], a, b, c) for j in indices if j not in (i_prev, i_curr, i_next)):
                continue
            triangles.append([i_prev, i_curr, i_next])
            indices.pop(i)
            ear_found = True
            break
        if not ear_found:
            break  # degenerate case — bail gracefully with what's clipped so far
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


def deform_idw(vertices, control_indices, control_targets, power=2.0, epsilon=1.0):
    vertices = np.asarray(vertices, dtype=float)
    deformed = vertices.copy()
    control_pts = vertices[control_indices]
    displacements = np.array(control_targets) - control_pts
    for i, v in enumerate(vertices):
        if i in control_indices:
            deformed[i] = control_targets[control_indices.index(i)]
            continue
        d = np.linalg.norm(control_pts - v, axis=1)
        w = 1.0 / (d ** power + epsilon)
        deformed[i] = v + (w[:, None] * displacements).sum(axis=0) / w.sum()
    return deformed


def render_mesh(vertices_rest, vertices_deformed, triangles, texture_arr, canvas_w, canvas_h):
    out = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    tex_h, tex_w = texture_arr.shape[:2]
    for tri in triangles:
        rest = vertices_rest[tri]
        dst = vertices_deformed[tri]
        x0, y0 = np.floor(dst.min(axis=0)).astype(int)
        x1, y1 = np.ceil(dst.max(axis=0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, canvas_w), min(y1, canvas_h)
        if x1 <= x0 or y1 <= y0:
            continue
        denom = (dst[1,1]-dst[2,1])*(dst[0,0]-dst[2,0]) + (dst[2,0]-dst[1,0])*(dst[0,1]-dst[2,1])
        if abs(denom) < 1e-9:
            continue
        for py in range(y0, y1):
            for px in range(x0, x1):
                p = np.array([px + 0.5, py + 0.5])
                u = ((dst[1,1]-dst[2,1])*(p[0]-dst[2,0]) + (dst[2,0]-dst[1,0])*(p[1]-dst[2,1])) / denom
                v = ((dst[2,1]-dst[0,1])*(p[0]-dst[2,0]) + (dst[0,0]-dst[2,0])*(p[1]-dst[2,1])) / denom
                w = 1 - u - v
                if u < 0 or v < 0 or w < 0:
                    continue
                src = u*rest[0] + v*rest[1] + w*rest[2]
                sx, sy = int(np.clip(src[0], 0, tex_w-1)), int(np.clip(src[1], 0, tex_h-1))
                out[py, px] = texture_arr[sy, sx]
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Building real AvatarRepresentation...")
    rep = build_avatar_representation()
    aw, ah = rep.image.size
    a_l_uy, _ = find_semantic_underarm(rep.torso_mask, rep.upper_arms_mask, aw, "left")
    a_r_uy, _ = find_semantic_underarm(rep.torso_mask, rep.upper_arms_mask, aw, "right")
    a_l_row = rep.torso_mask[a_l_uy]; a_l_xs = np.where(a_l_row[:aw//2])[0]
    a_left_ua = np.array([a_l_xs.min(), a_l_uy], dtype=float)
    a_r_row = rep.torso_mask[a_r_uy]; a_r_xs = np.where(a_r_row[aw//2:])[0] + aw//2
    a_right_ua = np.array([a_r_xs.max(), a_r_uy], dtype=float)
    a_left_sh = np.array(rep.left_shoulder, dtype=float)
    a_right_sh = np.array(rep.right_shoulder, dtype=float)

    print("Fetching real golden garment...")
    import requests
    garment_resp = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{GOLDEN_GARMENT_ID}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    garment_bytes = requests.get(garment_resp["clean_image_url"]).content
    mask, garment_img = get_binary_mask(garment_bytes)
    garment_arr = np.array(garment_img.convert("RGBA"))

    gw = garment_img.size[0]
    g_left_ua, g_right_ua = find_garment_underarms(mask, gw)
    g_left_sh, g_right_sh = get_garment_shoulder_points(mask, garment_img, g_left_ua, g_right_ua, gw)
    g_left_ua, g_right_ua = np.array(g_left_ua, dtype=float), np.array(g_right_ua, dtype=float)
    g_left_sh, g_right_sh = np.array(g_left_sh, dtype=float), np.array(g_right_sh, dtype=float)

    # Anatomical normalization — confirmed production rule: small-x -> right
    g_right_ua, g_left_ua = (g_left_ua, g_right_ua) if g_left_ua[0] < g_right_ua[0] else (g_right_ua, g_left_ua)
    g_right_sh, g_left_sh = (g_left_sh, g_right_sh) if g_left_sh[0] < g_right_sh[0] else (g_right_sh, g_left_sh)
    a_right_ua, a_left_ua = (a_left_ua, a_right_ua) if a_left_ua[0] < a_right_ua[0] else (a_right_ua, a_left_ua)
    a_right_sh, a_left_sh = (a_left_sh, a_right_sh) if a_left_sh[0] < a_right_sh[0] else (a_right_sh, a_left_sh)

    # ---------------- Δθ: the actual measurement ----------------
    theta_g = np.degrees(np.arctan2(g_left_ua[1]-g_right_ua[1], g_left_ua[0]-g_right_ua[0]))
    theta_a = np.degrees(np.arctan2(a_left_ua[1]-a_right_ua[1], a_left_ua[0]-a_right_ua[0]))
    delta_theta = theta_a - theta_g
    print(f"\nGarment underarm line angle  θ_g = {theta_g:.2f}°")
    print(f"Avatar underarm line angle   θ_a = {theta_a:.2f}°")
    print(f"Δθ = {delta_theta:.2f}°\n")

    # ---------------- A: REFERENCE — proven backend transform ----------------
    ref_matrix = solve_similarity_2point([g_right_ua, g_left_ua], [a_right_ua, a_left_ua])
    ref_full = cv2.warpAffine(garment_arr, ref_matrix, (aw, ah), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    ref_shoulder_left = ref_matrix @ np.array([*g_left_sh, 1.0])
    ref_shoulder_right = ref_matrix @ np.array([*g_right_sh, 1.0])

    avatar_rgba = np.array(rep.image.convert("RGBA"))
    ref_composite = avatar_rgba.copy()
    alpha = ref_full[:, :, 3:4].astype(float) / 255.0
    ref_composite[:, :, :3] = (avatar_rgba[:, :, :3] * (1 - alpha) + ref_full[:, :, :3] * alpha).astype(np.uint8)
    Image.fromarray(ref_composite).convert("RGB").save(f"{OUTPUT_DIR}/idw_rotation_test_REFERENCE.png")

    # ---------------- B: IDW on a boundary-only mesh ----------------
    print("Building boundary-only mesh from real garment silhouette (ear-clipping, no compiled dependency)...")
    vertices, triangles = build_boundary_mesh(mask)
    print(f"Mesh: {len(vertices)} boundary vertices, {len(triangles)} triangles")

    def nearest_idx(pt):
        return int(np.argmin(np.linalg.norm(vertices - pt, axis=1)))

    idx_right_ua, idx_left_ua = nearest_idx(g_right_ua), nearest_idx(g_left_ua)
    idx_right_sh, idx_left_sh = nearest_idx(g_right_sh), nearest_idx(g_left_sh)

    deformed = deform_idw(vertices, [idx_right_ua, idx_left_ua], [a_right_ua, a_left_ua])
    idw_shoulder_right = deformed[idx_right_sh]
    idw_shoulder_left = deformed[idx_left_sh]

    idw_layer = render_mesh(vertices, deformed, triangles, garment_arr, aw, ah)
    idw_composite = avatar_rgba.copy()
    alpha2 = idw_layer[:, :, 3:4].astype(float) / 255.0
    idw_composite[:, :, :3] = (avatar_rgba[:, :, :3] * (1 - alpha2) + idw_layer[:, :, :3] * alpha2).astype(np.uint8)
    Image.fromarray(idw_composite).convert("RGB").save(f"{OUTPUT_DIR}/idw_rotation_test_IDW.png")

    # ---------------- The actual comparison ----------------
    err_right_sh = np.linalg.norm(idw_shoulder_right - ref_shoulder_right[:2])
    err_left_sh = np.linalg.norm(idw_shoulder_left - ref_shoulder_left[:2])

    print(f"\n{'='*60}")
    print("SHOULDER POSITION — IDW (boundary-only mesh) vs REFERENCE (proven transform)")
    print(f"{'='*60}")
    print(f"Right shoulder:  REFERENCE={ref_shoulder_right[:2]}  IDW={idw_shoulder_right}  |  error={err_right_sh:.1f}px")
    print(f"Left shoulder:   REFERENCE={ref_shoulder_left[:2]}  IDW={idw_shoulder_left}  |  error={err_left_sh:.1f}px")
    print(f"\nFor scale: avatar underarm-to-underarm distance = {np.linalg.norm(a_left_ua - a_right_ua):.1f}px")
    print(f"\nSaved: {OUTPUT_DIR}/idw_rotation_test_REFERENCE.png")
    print(f"Saved: {OUTPUT_DIR}/idw_rotation_test_IDW.png")
    print(f"\n>>> DECISION GATE: inspect both images side by side. Is the shoulder")
    print(f">>> error above small relative to the underarm distance, and do the")
    print(f">>> two renders look equivalent? Or is there a visible, meaningful")
    print(f">>> orientation discrepancy in the IDW result?")


if __name__ == "__main__":
    main()