"""
tryon/checkpoint5_ARAP0_deformation_field.py

Three diagnostics on the EXACT SAME frozen ARAP-0 solve (same mesh,
same 4 handles, same uniform weights, same rest-position cold start,
same 60 iterations) — no changes to the mechanism itself:

  1. EDGE-LENGTH FIELD: rest-vs-deformed edge length ratio along the
     shortest mesh path from each shoulder handle to the topmost
     vertex (the collar point implicated in the head investigation).
     Distinguishes "ARAP preserved rest-scale rigidity far from
     handles" from "the solve degraded progressively along the path."

  2. INTERSECTION LOCALIZATION: for each of the (expected) 39 precise
     overlap pairs, reports which vertices are involved, their rest
     y-position (upper/lower in the original garment), and graph
     distance to the nearest handle — plus whether any of the 5
     inverted triangles participate in an overlap pair.

  3. ROTATION FIELD: for every mesh edge (i,j), the angular difference
     between R_i and R_j (both pure 2D rotations), to check whether
     the deformation field is smooth or has sharp local jumps, and
     where the largest jumps are.

Run: python -m tryon.checkpoint5_ARAP0_deformation_field
"""

import os
import ctypes
import heapq
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

# ---------------------------------------------------------------
# Identical mesh construction + ARAP solver — same code as every
# ARAP-0 script tonight. run_arap now ALSO returns the final
# rotations array (previously discarded) for the rotation-field check.
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
            adj[a].add(b); adj[b].add(a)
    return {i: sorted(v) for i, v in adj.items()}

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
            rotations[i] = np.eye(2); continue
        S = np.zeros((2, 2))
        for j in neighbors:
            S += np.outer(rest[i]-rest[j], current[i]-current[j])
        U, _, Vt = np.linalg.svd(S)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt_fixed = Vt.copy(); Vt_fixed[-1,:] *= -1
            R = Vt_fixed.T @ U.T
        rotations[i] = R
    return rotations

def global_step(L_factorized, rest, adj, rotations, n, handle_indices, handle_targets):
    bx, by = np.zeros(n), np.zeros(n)
    for i in range(n):
        if i in handle_indices:
            bx[i], by[i] = handle_targets[i]; continue
        acc = np.zeros(2)
        for j in adj[i]:
            acc += 0.5 * (rotations[i]+rotations[j]) @ (rest[i]-rest[j])
        bx[i], by[i] = acc
    return np.stack([L_factorized(bx), L_factorized(by)], axis=1)

def run_arap(rest, adj, handle_indices, handle_targets, max_iterations):
    n = len(rest)
    current = rest.copy()
    for h in handle_indices:
        current[h] = handle_targets[h]
    L = build_laplacian(n, adj, handle_indices)
    L_factorized = spla.factorized(L.tocsc())
    rotations = None
    for it in range(max_iterations):
        rotations = local_step(rest, current, adj, n)
        current = global_step(L_factorized, rest, adj, rotations, n, handle_indices, handle_targets)
    return current, rotations  # NOW returns final rotations too


# ---------------------------------------------------------------
# Precise overlap detector — same Sutherland-Hodgman as E/C/validation.
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
    if abs(denom) < 1e-12: return None
    t = ((p3[0]-p1[0])*d2y - (p3[1]-p1[1])*d2x) / denom
    return (p1[0]+t*d1x, p1[1]+t*d1y)
def clip_polygon(subject, clip):
    output = subject
    for i in range(len(clip)):
        if not output: break
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
    if len(poly) < 3: return 0.0
    return abs(sum(poly[i][0]*poly[(i+1)%len(poly)][1] - poly[(i+1)%len(poly)][0]*poly[i][1] for i in range(len(poly)))) / 2.0
def detect_overlap(a1, b1, c1, a2, b2, c2):
    return polygon_area(clip_polygon(ensure_ccw([a1,b1,c1]), ensure_ccw([a2,b2,c2])))


def main():
    print("Rebuilding real AvatarRepresentation and GarmentRepresentation...")
    avatar = build_avatar_representation()
    garment = build_garment_representation(GOLDEN_GARMENT_ID, AVATAR_USER_ID)

    vertices, triangles, landmark_index, n_boundary = build_dense_mesh(garment)
    n = len(vertices)

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
    shoulder_handle_indices = [landmark_index["left_shoulder"], landmark_index["right_shoulder"]]

    adj = build_adjacency(n, triangles)
    print(f"Re-running deterministic ARAP-0 solve ({MAX_ITERATIONS} iterations)...")
    deformed, rotations = run_arap(vertices, adj, handle_indices, handle_targets, MAX_ITERATIONS)
    max_residual = max(float(np.linalg.norm(deformed[i] - t)) for i, t in handle_targets.items())
    print(f"  Determinism check: max handle residual = {max_residual:.6f}px (expect 0.000000)\n")

    # ============================================================
    # 1. EDGE-LENGTH FIELD — shoulder handle -> topmost vertex (34)
    # ============================================================
    print(f"{'='*70}\n1. EDGE-LENGTH FIELD: shoulder handle -> collar vertex 34\n{'='*70}")
    TOPMOST_VERTEX = 34  # confirmed in the head investigation as the topmost deformed point

    def bfs_path(start, goal, adj):
        prev = {start: None}
        q = [start]
        while q:
            u = q.pop(0)
            if u == goal:
                break
            for v in adj[u]:
                if v not in prev:
                    prev[v] = u
                    q.append(v)
        path = [goal]
        while prev[path[-1]] is not None:
            path.append(prev[path[-1]])
        return path[::-1]

    for sh_idx in shoulder_handle_indices:
        path = bfs_path(sh_idx, TOPMOST_VERTEX, adj)
        print(f"\n  Path from shoulder handle {sh_idx} to vertex {TOPMOST_VERTEX} ({len(path)-1} hops):")
        for k in range(len(path)-1):
            i, j = path[k], path[k+1]
            rest_len = float(np.linalg.norm(vertices[i] - vertices[j]))
            def_len = float(np.linalg.norm(deformed[i] - deformed[j]))
            ratio = def_len / rest_len if rest_len > 1e-9 else float('nan')
            print(f"    edge ({i:3d}->{j:3d})  rest={rest_len:7.2f}  deformed={def_len:7.2f}  ratio={ratio:.3f}")
    print(f"\n  For reference, handle transform scales: T_underarm≈0.403, T_shoulder≈0.303")
    print(f"  (a ratio near these values = locally scaled correctly; a ratio near 1.0 = rest-scale preserved, NOT scaled)")

    # ============================================================
    # 2. INTERSECTION LOCALIZATION
    # ============================================================
    print(f"\n{'='*70}\n2. INTERSECTION LOCALIZATION (precise overlap pairs)\n{'='*70}")

    # Graph distance from nearest handle, for every vertex (multi-source BFS)
    dist_to_handle = {i: float('inf') for i in range(n)}
    q = []
    for h in handle_indices:
        dist_to_handle[h] = 0
        q.append(h)
    while q:
        u = q.pop(0)
        for v in adj[u]:
            if dist_to_handle[v] > dist_to_handle[u] + 1:
                dist_to_handle[v] = dist_to_handle[u] + 1
                q.append(v)

    ref_winding = None
    inverted_set = set()
    for ti, tri in enumerate(triangles):
        a, b, c = deformed[list(tri)]
        area2 = (b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])
        if abs(area2) < 1e-6:
            continue
        sign = 1 if area2 > 0 else -1
        if ref_winding is None:
            ref_winding = sign
        elif sign != ref_winding:
            inverted_set.add(ti)

    overlap_details = []
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
                rest_y_i = float(np.mean(vertices[list(ti)][:,1]))
                rest_y_j = float(np.mean(vertices[list(tj)][:,1]))
                min_hops = min(min(dist_to_handle[v] for v in ti), min(dist_to_handle[v] for v in tj))
                involves_inverted = (i in inverted_set) or (j in inverted_set)
                overlap_details.append((i, j, area, rest_y_i, rest_y_j, min_hops, involves_inverted))

    print(f"  Total precise overlap pairs: {len(overlap_details)}  (expect 39, from the validation run)")
    print(f"  Inverted triangles involved in at least one overlap: "
          f"{sum(1 for d in overlap_details if d[6])} / {len(overlap_details)} pairs")
    print(f"\n  Rest-y range of garment: {vertices[:,1].min():.1f} (top/collar) to {vertices[:,1].max():.1f} (hem)")
    ys_involved = [d[3] for d in overlap_details] + [d[4] for d in overlap_details]
    print(f"  Overlapping triangles' rest-y range: {min(ys_involved):.1f} to {max(ys_involved):.1f}")
    print(f"  Graph-hop distance to nearest handle, overlapping triangles: "
          f"min={min(d[5] for d in overlap_details)}  max={max(d[5] for d in overlap_details)}  "
          f"mean={np.mean([d[5] for d in overlap_details]):.1f}")
    print(f"\n  First 10 overlap pairs in detail:")
    for i, j, area, ry_i, ry_j, hops, inv in overlap_details[:10]:
        print(f"    tri{i} (rest_y={ry_i:.0f}) vs tri{j} (rest_y={ry_j:.0f})  area={area:.1f}px²  "
              f"min_hops_to_handle={hops}  involves_inverted={inv}")

    # ============================================================
    # 3. ROTATION FIELD
    # ============================================================
    print(f"\n{'='*70}\n3. ROTATION FIELD (angular difference between neighboring R_i)\n{'='*70}")

    def angle_of(R):
        return float(np.degrees(np.arctan2(R[1,0], R[0,0])))

    edge_rotation_diffs = []
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            ai, aj = angle_of(rotations[i]), angle_of(rotations[j])
            diff = abs(((ai - aj) + 180) % 360 - 180)
            edge_rotation_diffs.append((i, j, diff))

    diffs_only = [d[2] for d in edge_rotation_diffs]
    print(f"  Edges examined: {len(edge_rotation_diffs)}")
    print(f"  Rotation-angle difference across edges: min={min(diffs_only):.2f}°  "
          f"max={max(diffs_only):.2f}°  mean={np.mean(diffs_only):.2f}°  median={np.median(diffs_only):.2f}°")

    edge_rotation_diffs.sort(key=lambda d: -d[2])
    print(f"\n  10 largest rotation jumps between neighboring vertices:")
    for i, j, diff in edge_rotation_diffs[:10]:
        hops_i, hops_j = dist_to_handle[i], dist_to_handle[j]
        print(f"    edge ({i:3d},{j:3d})  Δangle={diff:6.2f}°  "
              f"rest=({tuple(np.round(vertices[i],1))},{tuple(np.round(vertices[j],1))})  "
              f"hops_to_handle=({hops_i},{hops_j})")

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"  Edge-length field: see ratios above vs transform scales (~0.30-0.40)")
    print(f"  Overlap pairs: {len(overlap_details)}, hop-distance range {min(d[5] for d in overlap_details)}-{max(d[5] for d in overlap_details)} from nearest handle")
    print(f"  Rotation field: max neighbor difference {max(diffs_only):.2f}°, mean {np.mean(diffs_only):.2f}°")


if __name__ == "__main__":
    main()