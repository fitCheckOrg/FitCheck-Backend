"""
tryon/checkpoint5_ARAP0.py

ARAP-0 — first real As-Rigid-As-Possible deformation test, fully
specified per the five locked decisions:

  1. Handle width:    single-vertex point handles
  2. Edge weights:    uniform (w_ij = 1 for every mesh edge)
  3. Initialization:  rest positions, only handles snapped to target
  4. Convergence:     energy-based, |ΔE|/E_prev, tolerance left OPEN —
                       this run does NOT stop early; it runs the full
                       iteration cap and logs the complete trace, so
                       the tolerance gets chosen from real data
  5. Scope:           4 handles (both underarms + both shoulders) only.
                       ARAP-0b (underarms-only) deliberately deferred.

Mesh: IDENTICAL construction to checkpoint4_C_geodesic_experiment.py —
same real poly2tri engine, same Poisson seed/params, same landmark
nudge logic — to guarantee the same 129-vertex/218-triangle topology
E and C both used.

Handle rest positions are the EXISTING TopMesh vertices (e.g. the
shoulder's ~1px discrepancy from the raw semantic landmark is
preserved, never pre-corrected). Handle targets come from
AvatarAsset.landmarks via the same coordinate-pairing convention
used throughout (pair by x-position, not by field name).

Algorithm: standard Sorkine & Alexa (2007) local/global ARAP.
  Local step:  per-vertex rotation via SVD of the neighborhood
               covariance (rest edges vs current edges).
  Global step: sparse Laplacian solve, handle rows replaced with
               identity + target RHS (hard constraints), matrix
               pre-factorized once since weights never change.

Requires: pip install scipy (if not already present)

Run: python -m tryon.checkpoint5_ARAP0
"""

import os
import ctypes
import numpy as np
import cv2
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from PIL import Image, ImageDraw

from tryon.avatar_representation import build_avatar_representation
from tryon.garment_representation import build_garment_representation

GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
OUTPUT_DIR = "outputs"
DLL_PATH = os.path.join(os.path.dirname(__file__), "p2t_minimal.dll")

POISSON_MIN_SPACING = 25.0
POISSON_MIN_CLEARANCE = 25.0
POISSON_SEED = 1234
POISSON_MAX_ATTEMPTS = 8000
STEINER_MIN_CLEARANCE = 1.0

MAX_ITERATIONS = 60  # safety-net cap (Decision 4B) — this run always runs to the cap,
                      # no early stop, so the full trace is available for choosing τ.

# ---------------------------------------------------------------
# Mesh construction — IDENTICAL to checkpoint4_C_geodesic_experiment.py
# ---------------------------------------------------------------
_lib = ctypes.CDLL(DLL_PATH)
_lib.p2t_create.restype = ctypes.c_void_p
_lib.p2t_create.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int32]
_lib.p2t_add_points.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int32]
_lib.p2t_triangulate.argtypes = [ctypes.c_void_p]
_lib.p2t_get_triangle_count.argtypes = [ctypes.c_void_p]
_lib.p2t_get_triangles.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
_lib.p2t_destroy.argtypes = [ctypes.c_void_p]


def real_poly2tri_triangulate(boundary, interior):
    n_b, n_i = len(boundary), len(interior)
    b_arr = (ctypes.c_double * (n_b * 2))(*np.array(boundary).flatten())
    ctx = _lib.p2t_create(b_arr, n_b)
    if not ctx:
        raise RuntimeError("p2t_create failed")
    if n_i > 0:
        i_arr = (ctypes.c_double * (n_i * 2))(*np.array(interior).flatten())
        rc = _lib.p2t_add_points(ctx, i_arr, n_i)
        if rc != 0:
            raise RuntimeError(f"p2t_add_points failed, rc={rc}")
    rc = _lib.p2t_triangulate(ctx)
    if rc != 0:
        raise RuntimeError(f"p2t_triangulate failed, rc={rc}")
    count = _lib.p2t_get_triangle_count(ctx)
    buf = (ctypes.c_int32 * (count * 3))()
    _lib.p2t_get_triangles(ctx, buf, count * 3)
    _lib.p2t_destroy(ctx)
    return [tuple(buf[3*i:3*i+3]) for i in range(count)]


def trace_and_simplify(mask, epsilon=2.0):
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    raw = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    approx = cv2.approxPolyDP(raw.reshape(-1, 1, 2).astype(np.float32), epsilon, closed=True)
    boundary = approx.reshape(-1, 2)
    if np.allclose(boundary[0], boundary[-1]):
        boundary = boundary[:-1]
    return [tuple(p) for p in boundary]


def dist_to_segment(p, a, b):
    p, a, b = np.array(p, float), np.array(a, float), np.array(b, float)
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def min_dist_to_boundary(p, boundary):
    n = len(boundary)
    return min(dist_to_segment(p, boundary[i], boundary[(i+1) % n]) for i in range(n))


def point_in_polygon(p, poly):
    p = (float(p[0]), float(p[1]))
    poly = np.array([[float(x), float(y)] for x, y in poly], dtype=np.float32)
    return cv2.pointPolygonTest(poly, p, False) >= 0


def nudge_inward(point, boundary, distance):
    best_dist, best_seg, best_q = float('inf'), None, None
    n = len(boundary)
    for i in range(n):
        a, b = np.array(boundary[i]), np.array(boundary[(i+1) % n])
        d = dist_to_segment(point, a, b)
        if d < best_dist:
            ab = b - a
            len_sq = np.dot(ab, ab)
            t = 0.0 if len_sq < 1e-12 else np.clip(np.dot(np.array(point)-a, ab)/len_sq, 0, 1)
            best_dist, best_seg, best_q = d, (a, b), a + t*ab
    a, b = best_seg
    seg_dir = (b - a) / (np.linalg.norm(b - a) + 1e-12)
    n1 = np.array([-seg_dir[1], seg_dir[0]])
    n2 = -n1
    test_eps = 0.1
    p1, p2 = best_q + n1*test_eps, best_q + n2*test_eps
    if point_in_polygon(p1, boundary):
        inward = n1
    elif point_in_polygon(p2, boundary):
        inward = n2
    else:
        raise RuntimeError("Neither perpendicular direction tested inward")
    return tuple(best_q + inward*distance)


def poisson_interior(boundary, extra_seeds, min_spacing, min_clearance, seed, max_attempts, w, h):
    rng = np.random.RandomState(seed)
    accepted = list(boundary) + list(extra_seeds)
    out = []
    for _ in range(max_attempts):
        c = (rng.uniform(0, w), rng.uniform(0, h))
        if not point_in_polygon(c, boundary):
            continue
        if min_dist_to_boundary(c, boundary) < min_clearance:
            continue
        if any(np.hypot(c[0]-p[0], c[1]-p[1]) < min_spacing for p in accepted):
            continue
        accepted.append(c)
        out.append(c)
    return out


def build_dense_mesh(garment):
    mask = garment.mask
    h, w = mask.shape
    boundary = trace_and_simplify(mask, epsilon=2.0)
    n_boundary = len(boundary)

    landmark_pts = {
        "left_underarm": garment.left_underarm, "right_underarm": garment.right_underarm,
        "left_shoulder": garment.left_shoulder, "right_shoulder": garment.right_shoulder,
    }
    landmark_index = {}
    unmatched = {}
    for name, pt in landmark_pts.items():
        match = None
        for i, bp in enumerate(boundary):
            if np.hypot(pt[0]-bp[0], pt[1]-bp[1]) < 4.0:
                match = i
                break
        if match is not None:
            landmark_index[name] = match
        else:
            unmatched[name] = pt

    steiner_pts = []
    idx = n_boundary
    for name, pt in unmatched.items():
        needs_nudge = (not point_in_polygon(pt, boundary)) or (min_dist_to_boundary(pt, boundary) < STEINER_MIN_CLEARANCE)
        final_pt = nudge_inward(pt, boundary, STEINER_MIN_CLEARANCE) if needs_nudge else pt
        if needs_nudge:
            print(f"  Nudged {name}: {pt} -> {final_pt}")
        steiner_pts.append(final_pt)
        landmark_index[name] = idx
        idx += 1

    poisson_pts = poisson_interior(boundary, steiner_pts, POISSON_MIN_SPACING, POISSON_MIN_CLEARANCE,
                                    POISSON_SEED, POISSON_MAX_ATTEMPTS, w, h)
    print(f"  Poisson interior: {len(poisson_pts)} points")

    all_vertices = boundary + steiner_pts + poisson_pts
    triangles = real_poly2tri_triangulate(boundary, steiner_pts + poisson_pts)
    print(f"  Triangulated: {len(triangles)} triangles (must read 218 to confirm same mesh as E/C)")

    return np.array(all_vertices), triangles, landmark_index, n_boundary


def build_adjacency(n, triangles):
    """Uniform weights (Decision 2): every edge weight = 1."""
    adj = {i: set() for i in range(n)}
    for tri in triangles:
        for a, b in [(tri[0],tri[1]), (tri[1],tri[2]), (tri[2],tri[0])]:
            adj[a].add(b)
            adj[b].add(a)
    return {i: sorted(neighbors) for i, neighbors in adj.items()}


# ---------------------------------------------------------------
# ARAP solver — standard Sorkine & Alexa (2007) local/global.
# ---------------------------------------------------------------

def build_laplacian(n, adj, handle_indices):
    """Uniform-weight graph Laplacian, with handle rows replaced by
    identity (hard constraint via Dirichlet boundary condition)."""
    rows, cols, vals = [], [], []
    for i in range(n):
        if i in handle_indices:
            rows.append(i); cols.append(i); vals.append(1.0)
            continue
        deg = len(adj[i])
        rows.append(i); cols.append(i); vals.append(float(deg))
        for j in adj[i]:
            rows.append(i); cols.append(j); vals.append(-1.0)
    L = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return L


def local_step(rest, current, adj, n):
    """Per-vertex optimal rotation via SVD (Sorkine & Alexa eq. 5-6)."""
    rotations = np.zeros((n, 2, 2))
    for i in range(n):
        neighbors = adj[i]
        if not neighbors:
            rotations[i] = np.eye(2)
            continue
        S = np.zeros((2, 2))
        for j in neighbors:
            e_rest = rest[i] - rest[j]
            e_cur = current[i] - current[j]
            S += np.outer(e_rest, e_cur)  # uniform weight = 1
        U, _, Vt = np.linalg.svd(S)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt_fixed = Vt.copy()
            Vt_fixed[-1, :] *= -1
            R = Vt_fixed.T @ U.T
        rotations[i] = R
    return rotations


def global_step(L_factorized, rest, adj, rotations, n, handle_indices, handle_targets):
    """Solve the prefactorized Laplacian system for both x and y."""
    bx = np.zeros(n)
    by = np.zeros(n)
    for i in range(n):
        if i in handle_indices:
            bx[i], by[i] = handle_targets[i]
            continue
        acc = np.zeros(2)
        for j in adj[i]:
            e_rest = rest[i] - rest[j]
            acc += 0.5 * (rotations[i] + rotations[j]) @ e_rest
        bx[i], by[i] = acc
    new_x = L_factorized(bx)
    new_y = L_factorized(by)
    return np.stack([new_x, new_y], axis=1)


def compute_energy(rest, current, rotations, adj, n):
    """E = Σ_i Σ_j w_ij ||(p'_i - p'_j) - R_i(p_i - p_j)||^2, uniform w_ij=1."""
    total = 0.0
    for i in range(n):
        for j in adj[i]:
            e_rest = rest[i] - rest[j]
            e_cur = current[i] - current[j]
            diff = e_cur - rotations[i] @ e_rest
            total += float(np.dot(diff, diff))
    return total


def run_arap(rest, adj, handle_indices, handle_targets, max_iterations):
    n = len(rest)
    current = rest.copy()
    for h in handle_indices:
        current[h] = handle_targets[h]

    L = build_laplacian(n, adj, handle_indices)
    L_factorized = spla.factorized(L.tocsc())  # factorize ONCE — weights never change

    trace = []
    prev_energy = None
    for it in range(1, max_iterations + 1):
        rotations = local_step(rest, current, adj, n)
        new_current = global_step(L_factorized, rest, adj, rotations, n, handle_indices, handle_targets)
        max_disp = float(np.max(np.linalg.norm(new_current - current, axis=1)))
        current = new_current
        energy = compute_energy(rest, current, rotations, adj, n)
        rel_change = abs(energy - prev_energy) / max(prev_energy, 1e-9) if prev_energy is not None else float('nan')
        trace.append({"iteration": it, "energy": energy, "relative_change": rel_change, "max_displacement": max_disp})
        print(f"  iter {it:3d}   E={energy:14.4f}   ΔE_rel={rel_change if it>1 else float('nan'):.6f}   max_disp={max_disp:.4f}px")
        prev_energy = energy

    return current, trace


def render_mesh(vertices_rest, vertices_deformed, triangles, texture_arr, canvas_w, canvas_h):
    out = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    tex_h, tex_w = texture_arr.shape[:2]
    for tri in triangles:
        rest, dst = vertices_rest[list(tri)], vertices_deformed[list(tri)]
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

    print("\nBuilding dense mesh (IDENTICAL construction to E/C)...")
    vertices, triangles, landmark_index, n_boundary = build_dense_mesh(garment)
    n = len(vertices)

    # Pair by coordinate (not field name) — same convention used throughout.
    g_ua = (garment.left_underarm, garment.right_underarm) if garment.left_underarm[0] < garment.right_underarm[0] else (garment.right_underarm, garment.left_underarm)
    g_sh = (garment.left_shoulder, garment.right_shoulder) if garment.left_shoulder[0] < garment.right_shoulder[0] else (garment.right_shoulder, garment.left_shoulder)
    a_ua = (avatar.left_underarm, avatar.right_underarm) if avatar.left_underarm[0] < avatar.right_underarm[0] else (avatar.right_underarm, avatar.left_underarm)
    a_sh = (avatar.left_shoulder, avatar.right_shoulder) if avatar.left_shoulder[0] < avatar.right_shoulder[0] else (avatar.right_shoulder, avatar.left_shoulder)

    # Map each landmark_index entry to its correctly-paired avatar target.
    handle_targets = {}
    for g_name, g_pt, a_pt in [("left_underarm" if garment.left_underarm[0]<garment.right_underarm[0] else "right_underarm", g_ua[0], a_ua[0]),
                                ("right_underarm" if garment.left_underarm[0]<garment.right_underarm[0] else "left_underarm", g_ua[1], a_ua[1]),
                                ("left_shoulder" if garment.left_shoulder[0]<garment.right_shoulder[0] else "right_shoulder", g_sh[0], a_sh[0]),
                                ("right_shoulder" if garment.left_shoulder[0]<garment.right_shoulder[0] else "left_shoulder", g_sh[1], a_sh[1])]:
        handle_targets[landmark_index[g_name]] = np.array(a_pt, dtype=float)

    handle_indices = set(handle_targets.keys())
    print(f"\nHandle vertices: {handle_indices}")
    for idx, target in handle_targets.items():
        print(f"  vertex {idx}: rest={tuple(vertices[idx])}  ->  target={tuple(target)}")

    adj = build_adjacency(n, triangles)

    print(f"\n{'='*70}\nRUNNING ARAP-0 — {MAX_ITERATIONS} iterations, full trace, no early stop\n{'='*70}")
    deformed, trace = run_arap(vertices, adj, handle_indices, handle_targets, MAX_ITERATIONS)

    # Handle residuals (should be exactly 0 — hard constraints).
    print(f"\n{'='*70}\nHANDLE RESIDUALS (should be 0.0000px — hard constraints)\n{'='*70}")
    for idx, target in handle_targets.items():
        err = float(np.linalg.norm(deformed[idx] - target))
        print(f"  vertex {idx}: error={err:.6f}px")

    # Structural integrity
    print(f"\n{'='*70}\nMESH STRUCTURAL INTEGRITY\n{'='*70}")
    ref_winding = None
    inverted, degenerate = 0, 0
    for tri in triangles:
        a, b, c = deformed[list(tri)]
        area2 = (b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])
        if abs(area2) < 1e-6:
            degenerate += 1
            continue
        sign = 1 if area2 > 0 else -1
        if ref_winding is None:
            ref_winding = sign
        elif sign != ref_winding:
            inverted += 1
    print(f"  Inverted: {inverted} / {len(triangles)}")
    print(f"  Degenerate: {degenerate} / {len(triangles)}")

    overlap_pairs = 0
    for i in range(len(triangles)):
        ti = triangles[i]
        for j in range(i+1, len(triangles)):
            tj = triangles[j]
            if set(ti) & set(tj):
                continue
            # cheap bbox-overlap pre-filter, then real check only if bboxes intersect
            a1 = deformed[list(ti)]; a2 = deformed[list(tj)]
            if (a1[:,0].max() < a2[:,0].min() or a2[:,0].max() < a1[:,0].min() or
                a1[:,1].max() < a2[:,1].min() or a2[:,1].max() < a1[:,1].min()):
                continue
            overlap_pairs += 1  # bbox overlap flagged; coarse signal, not the precise Sutherland-Hodgman count
    print(f"  Bounding-box overlap candidates (coarse signal): {overlap_pairs}")

    # Hem / sleeve diagnostic points — same 5 points as E/C.
    print(f"\n{'='*70}\nHEM AND SLEEVE BEHAVIOR\n{'='*70}")
    mask = garment.mask
    ys, xs = np.where(mask)
    bottom_y = ys.max()
    bottom_xs = xs[ys == bottom_y]
    test_points = {
        "left_hem": (float(bottom_xs.min()), float(bottom_y)),
        "center_hem": (float(np.median(bottom_xs)), float(bottom_y)),
        "right_hem": (float(bottom_xs.max()), float(bottom_y)),
        "left_sleeve_end": (float(xs.min()), float(ys[xs==xs.min()].mean())),
        "right_sleeve_end": (float(xs.max()), float(ys[xs==xs.max()].mean())),
    }
    for name, pt in test_points.items():
        nearest = int(np.argmin(np.linalg.norm(vertices - np.array(pt), axis=1)))
        print(f"  {name:18s} nearest_vertex={nearest}  rest={tuple(vertices[nearest])}  -> deformed={tuple(deformed[nearest])}")

    # Render
    garment_arr = np.array(garment.image.convert("RGBA"))
    avatar_rgba = np.array(avatar.image.convert("RGBA"))
    aw, ah = avatar.image.size
    layer = render_mesh(vertices, deformed, triangles, garment_arr, aw, ah)
    composite = avatar_rgba.copy()
    alpha = layer[:,:,3:4].astype(float)/255.0
    composite[:,:,:3] = (avatar_rgba[:,:,:3]*(1-alpha) + layer[:,:,:3]*alpha).astype(np.uint8)
    Image.fromarray(composite).convert("RGB").save(f"{OUTPUT_DIR}/checkpoint5_ARAP0_render.png")

    annotated = Image.fromarray(composite).convert("RGB")
    draw = ImageDraw.Draw(annotated)
    colors = {"left_hem":(255,0,0),"center_hem":(255,128,0),"right_hem":(255,0,255),
              "left_sleeve_end":(0,255,255),"right_sleeve_end":(0,255,0)}
    for name, pt in test_points.items():
        nearest = int(np.argmin(np.linalg.norm(vertices - np.array(pt), axis=1)))
        x, y = deformed[nearest]
        draw.ellipse([x-6,y-6,x+6,y+6], outline=colors[name], width=3)
    for target in handle_targets.values():
        x, y = target
        draw.ellipse([x-5,y-5,x+5,y+5], outline=(255,255,0), width=2)
    annotated.save(f"{OUTPUT_DIR}/checkpoint5_ARAP0_render_annotated.png")

    print(f"\nSaved: {OUTPUT_DIR}/checkpoint5_ARAP0_render.png")
    print(f"Saved: {OUTPUT_DIR}/checkpoint5_ARAP0_render_annotated.png")
    print(f"\n>>> Full convergence trace printed above — choose τ by inspecting where")
    print(f">>> relative_change actually flattens, don't assume 0.1% or any other value.")


if __name__ == "__main__":
    main()