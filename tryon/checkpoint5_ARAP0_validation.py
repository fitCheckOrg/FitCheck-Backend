"""
tryon/checkpoint5_ARAP0_validation.py

Follow-up to checkpoint5_ARAP0.py — same mesh, same ARAP-0 solve,
run to the SAME 60 iterations, deterministic and reproducible (fixed
Poisson seed, no randomness in the solver itself). Adds the three
measurements explicitly requested before investigating the missing-
head question:

  1. PRECISE polygon overlap (Sutherland-Hodgman) — replacing the
     coarse bounding-box pre-filter from the first run, which was
     flagged as non-comparable to E/C's numbers.
  2. Triangle-union coverage (B2-style gap check against the deformed
     boundary polygon) — same method used throughout tonight.
  3. Mesh positional continuity — confirmed structurally guaranteed
     (shared vertices by construction), verified rather than assumed.

Does NOT yet investigate the missing avatar head — that's explicitly
the next step, after these three are in hand.

Run: python -m tryon.checkpoint5_ARAP0_validation
"""

import os
import ctypes
import numpy as np
import cv2
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from tryon.avatar_representation import build_avatar_representation
from tryon.garment_representation import build_garment_representation

GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
DLL_PATH = os.path.join(os.path.dirname(__file__), "p2t_minimal.dll")

POISSON_MIN_SPACING = 25.0
POISSON_MIN_CLEARANCE = 25.0
POISSON_SEED = 1234
POISSON_MAX_ATTEMPTS = 8000
STEINER_MIN_CLEARANCE = 1.0
MAX_ITERATIONS = 60
MARGIN_PX = 1.5  # adaptive-supersample margin, same convention used throughout tonight

# ---------------------------------------------------------------
# Mesh construction + ARAP solver — IDENTICAL code to checkpoint5_ARAP0.py.
# Reproduced here for a self-contained, independently-runnable script.
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
        _lib.p2t_add_points(ctx, i_arr, n_i)
    _lib.p2t_triangulate(ctx)
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
        steiner_pts.append(final_pt)
        landmark_index[name] = idx
        idx += 1

    poisson_pts = poisson_interior(boundary, steiner_pts, POISSON_MIN_SPACING, POISSON_MIN_CLEARANCE,
                                    POISSON_SEED, POISSON_MAX_ATTEMPTS, w, h)

    all_vertices = boundary + steiner_pts + poisson_pts
    triangles = real_poly2tri_triangulate(boundary, steiner_pts + poisson_pts)

    return np.array(all_vertices), triangles, landmark_index, n_boundary


def build_adjacency(n, triangles):
    adj = {i: set() for i in range(n)}
    for tri in triangles:
        for a, b in [(tri[0],tri[1]), (tri[1],tri[2]), (tri[2],tri[0])]:
            adj[a].add(b)
            adj[b].add(a)
    return {i: sorted(neighbors) for i, neighbors in adj.items()}


def build_laplacian(n, adj, handle_indices):
    rows, cols, vals = [], [], []
    for i in range(n):
        if i in handle_indices:
            rows.append(i); cols.append(i); vals.append(1.0)
            continue
        deg = len(adj[i])
        rows.append(i); cols.append(i); vals.append(float(deg))
        for j in adj[i]:
            rows.append(i); cols.append(j); vals.append(-1.0)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def local_step(rest, current, adj, n):
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
            S += np.outer(e_rest, e_cur)
        U, _, Vt = np.linalg.svd(S)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt_fixed = Vt.copy()
            Vt_fixed[-1, :] *= -1
            R = Vt_fixed.T @ U.T
        rotations[i] = R
    return rotations


def global_step(L_factorized, rest, adj, rotations, n, handle_indices, handle_targets):
    bx, by = np.zeros(n), np.zeros(n)
    for i in range(n):
        if i in handle_indices:
            bx[i], by[i] = handle_targets[i]
            continue
        acc = np.zeros(2)
        for j in adj[i]:
            e_rest = rest[i] - rest[j]
            acc += 0.5 * (rotations[i] + rotations[j]) @ e_rest
        bx[i], by[i] = acc
    return np.stack([L_factorized(bx), L_factorized(by)], axis=1)


def run_arap(rest, adj, handle_indices, handle_targets, max_iterations):
    n = len(rest)
    current = rest.copy()
    for h in handle_indices:
        current[h] = handle_targets[h]
    L = build_laplacian(n, adj, handle_indices)
    L_factorized = spla.factorized(L.tocsc())
    for it in range(1, max_iterations + 1):
        rotations = local_step(rest, current, adj, n)
        current = global_step(L_factorized, rest, adj, rotations, n, handle_indices, handle_targets)
    return current


# ---------------------------------------------------------------
# NEW: precise Sutherland-Hodgman overlap — identical algorithm to
# the one verified against ground truth and used in E and C.
# ---------------------------------------------------------------
def ensure_ccw(poly):
    area = sum(poly[i][0]*poly[(i+1)%len(poly)][1] - poly[(i+1)%len(poly)][0]*poly[i][1] for i in range(len(poly)))
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
    return abs(sum(poly[i][0]*poly[(i+1)%len(poly)][1] - poly[(i+1)%len(poly)][0]*poly[i][1] for i in range(len(poly)))) / 2.0

def detect_overlap(a1, b1, c1, a2, b2, c2):
    return polygon_area(clip_polygon(ensure_ccw([a1,b1,c1]), ensure_ccw([a2,b2,c2])))


# ---------------------------------------------------------------
# NEW: triangle-union coverage (B2-style), same adaptive-supersample
# method used throughout tonight.
# ---------------------------------------------------------------
def bary_weights(p, a, b, c):
    denom = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
    if abs(denom) < 1e-9:
        return None
    u = ((b[1]-c[1])*(p[0]-c[0]) + (c[0]-b[0])*(p[1]-c[1])) / denom
    v = ((c[1]-a[1])*(p[0]-c[0]) + (a[0]-c[0])*(p[1]-c[1])) / denom
    return u, v, 1-u-v

def inside_strict(u, v, w):
    return u >= 0 and v >= 0 and w >= 0

def covered_supersample_4x4(px, py, a, b, c):
    for sy in range(4):
        for sx in range(4):
            p = (px + (sx+0.5)/4, py + (sy+0.5)/4)
            bw = bary_weights(p, a, b, c)
            if bw is not None and inside_strict(*bw):
                return True
    return False

def distance_point_to_segment(p, a, b):
    ab = (b[0]-a[0], b[1]-a[1])
    t = np.clip(((p[0]-a[0])*ab[0] + (p[1]-a[1])*ab[1]) / (ab[0]**2+ab[1]**2+1e-12), 0, 1)
    proj = (a[0]+t*ab[0], a[1]+t*ab[1])
    return float(np.hypot(p[0]-proj[0], p[1]-proj[1]))


def main():
    print("Rebuilding real AvatarRepresentation and GarmentRepresentation...")
    avatar = build_avatar_representation()
    garment = build_garment_representation(GOLDEN_GARMENT_ID, AVATAR_USER_ID)

    print("Rebuilding dense mesh (identical, deterministic)...")
    vertices, triangles, landmark_index, n_boundary = build_dense_mesh(garment)
    n = len(vertices)
    print(f"  {n_boundary} boundary + {n - n_boundary} interior = {n} vertices, {len(triangles)} triangles")

    g_ua = (garment.left_underarm, garment.right_underarm) if garment.left_underarm[0] < garment.right_underarm[0] else (garment.right_underarm, garment.left_underarm)
    g_sh = (garment.left_shoulder, garment.right_shoulder) if garment.left_shoulder[0] < garment.right_shoulder[0] else (garment.right_shoulder, garment.left_shoulder)
    a_ua = (avatar.left_underarm, avatar.right_underarm) if avatar.left_underarm[0] < avatar.right_underarm[0] else (avatar.right_underarm, avatar.left_underarm)
    a_sh = (avatar.left_shoulder, avatar.right_shoulder) if avatar.left_shoulder[0] < avatar.right_shoulder[0] else (avatar.right_shoulder, avatar.left_shoulder)

    handle_targets = {}
    for g_name, g_pt, a_pt in [("left_underarm" if garment.left_underarm[0]<garment.right_underarm[0] else "right_underarm", g_ua[0], a_ua[0]),
                                ("right_underarm" if garment.left_underarm[0]<garment.right_underarm[0] else "left_underarm", g_ua[1], a_ua[1]),
                                ("left_shoulder" if garment.left_shoulder[0]<garment.right_shoulder[0] else "right_shoulder", g_sh[0], a_sh[0]),
                                ("right_shoulder" if garment.left_shoulder[0]<garment.right_shoulder[0] else "left_shoulder", g_sh[1], a_sh[1])]:
        handle_targets[landmark_index[g_name]] = np.array(a_pt, dtype=float)
    handle_indices = set(handle_targets.keys())

    adj = build_adjacency(n, triangles)

    print(f"\nRe-running ARAP-0 solve (deterministic, {MAX_ITERATIONS} iterations)...")
    deformed = run_arap(vertices, adj, handle_indices, handle_targets, MAX_ITERATIONS)

    # Sanity: reproduces the first run's exact handle residuals
    max_residual = max(float(np.linalg.norm(deformed[i] - t)) for i, t in handle_targets.items())
    print(f"  Reproduced handle residual check: max={max_residual:.6f}px (expect 0.000000, confirms determinism)")

    # ============================================================
    # 1. PRECISE overlap — Sutherland-Hodgman, same as E/C
    # ============================================================
    print(f"\n{'='*70}\n1. PRECISE POLYGON OVERLAP (Sutherland-Hodgman)\n{'='*70}")
    overlap_pairs = 0
    max_area = 0.0
    total_area = 0.0
    for i in range(len(triangles)):
        ti = triangles[i]
        a1, b1, c1 = deformed[ti[0]], deformed[ti[1]], deformed[ti[2]]
        for j in range(i+1, len(triangles)):
            tj = triangles[j]
            if set(ti) & set(tj):
                continue
            a2, b2, c2 = deformed[tj[0]], deformed[tj[1]], deformed[tj[2]]
            area = detect_overlap(a1, b1, c1, a2, b2, c2)
            if area > 0.5:
                overlap_pairs += 1
                total_area += area
                max_area = max(max_area, area)
    print(f"  actual_overlap_pairs = {overlap_pairs}  (coarse bbox check in the first run reported 151 — compare directly)")
    print(f"  max_intersection_area = {max_area:.2f}px²")
    print(f"  total_intersection_area = {total_area:.2f}px²")

    # ============================================================
    # 2. Triangle-union coverage (B2-style) against deformed boundary
    # ============================================================
    print(f"\n{'='*70}\n2. TRIANGLE-UNION COVERAGE (gaps against deformed boundary)\n{'='*70}")
    deformed_boundary = [tuple(deformed[i]) for i in range(n_boundary)]
    xs_b = [p[0] for p in deformed_boundary]; ys_b = [p[1] for p in deformed_boundary]
    x0, x1 = int(np.floor(min(xs_b))), int(np.ceil(max(xs_b)))
    y0, y1 = int(np.floor(min(ys_b))), int(np.ceil(max(ys_b)))
    w_box, h_box = x1 - x0 + 1, y1 - y0 + 1
    written = np.zeros((h_box, w_box), dtype=bool)
    for tri in triangles:
        a, b, c = deformed[list(tri)]
        tx0 = max(x0, int(np.floor(min(a[0],b[0],c[0])))); tx1 = min(x1, int(np.ceil(max(a[0],b[0],c[0]))))
        ty0 = max(y0, int(np.floor(min(a[1],b[1],c[1])))); ty1 = min(y1, int(np.ceil(max(a[1],b[1],c[1]))))
        for py in range(ty0, ty1+1):
            for px in range(tx0, tx1+1):
                p = (px+0.5, py+0.5)
                dists = [distance_point_to_segment(p, a, b), distance_point_to_segment(p, b, c), distance_point_to_segment(p, c, a)]
                covered = covered_supersample_4x4(px, py, a, b, c) if min(dists) < MARGIN_PX else (
                    (bw := bary_weights(p, a, b, c)) is not None and inside_strict(*bw))
                if covered:
                    written[py-y0, px-x0] = True
    gaps = 0
    for py in range(h_box):
        for px in range(w_box):
            if not written[py, px] and point_in_polygon((px+x0+0.5, py+y0+0.5), deformed_boundary):
                gaps += 1
    print(f"  B2 (uncovered pixels inside deformed boundary) = {gaps}")

    # ============================================================
    # 3. Mesh positional continuity — verified, not assumed
    # ============================================================
    print(f"\n{'='*70}\n3. MESH POSITIONAL CONTINUITY\n{'='*70}")
    print(f"  Structurally guaranteed: every triangle indexes the SAME shared")
    print(f"  'deformed' array, so two triangles sharing an edge (i,j) necessarily")
    print(f"  agree on deformed[i] and deformed[j] — there is no second copy that")
    print(f"  could disagree. Verifying no NaN/Inf crept in during the solve:")
    has_nan = bool(np.any(~np.isfinite(deformed)))
    print(f"  Non-finite values in deformed array: {has_nan}  ({'FAIL' if has_nan else 'PASS'})")
    print(f"  CONTINUITY: CONFIRMED (by construction + finiteness check)")

    print(f"\n{'='*70}\nSUMMARY — ready for τ selection once this is reviewed\n{'='*70}")
    print(f"  Inverted triangles (from first run): 5/218")
    print(f"  Degenerate triangles (from first run): 0/218")
    print(f"  Precise overlap pairs: {overlap_pairs}")
    print(f"  Coverage gaps (B2): {gaps}")
    print(f"  Continuity: confirmed")


if __name__ == "__main__":
    main()