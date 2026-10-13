"""
tryon/checkpoint4_C_geodesic_experiment.py

CHECKPOINT 4 — C-minimal: same E transform field, same weighting
formula, same overrides — only the distance metric changes, from
Euclidean segment distance to mesh-graph (geodesic) distance.

INPUTS: real GarmentRepresentation.mask / .left_underarm / etc. —
identical objects E used. See the verification table in chat for
exactly which downstream steps are and are not bit-identical to the
Flutter TopMeshGenerator pipeline (contour tracer and Poisson PRNG
are NOT identical; poly2tri engine, landmark nudge, and vertex
ordering ARE).

T_underarm, T_shoulder, POWER, EPSILON, and the landmark-override
logic are copied verbatim from checkpoint4_experiment_E.py — frozen,
not retuned, per the one-variable discipline.

Run: python -m tryon.checkpoint4_C_geodesic_experiment
"""

import os
import ctypes
import heapq
import numpy as np
import cv2
from PIL import Image, ImageDraw

from tryon.avatar_representation import build_avatar_representation
from tryon.garment_representation import build_garment_representation
from tryon.piecewise_continuous_test import solve_similarity_2point

GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
OUTPUT_DIR = "outputs"
DLL_PATH = os.path.join(os.path.dirname(__file__), "p2t_minimal.dll")

# Frozen from E — do not retune.
POWER = 2.0
EPSILON = 1.0
LANDMARK_OVERRIDE_RADIUS = 0.5

# Dense mesh parameters — matching Dart's TopMeshGenerator defaults.
POISSON_MIN_SPACING = 25.0
POISSON_MIN_CLEARANCE = 25.0
POISSON_SEED = 1234
POISSON_MAX_ATTEMPTS = 8000
STEINER_MIN_CLEARANCE = 1.0


# ---------------------------------------------------------------
# ctypes bindings to the real, freshly-compiled, Wine-verified
# poly2tri engine.
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
        raise RuntimeError("p2t_create failed — check boundary has no duplicate/closing point")
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
    rc = _lib.p2t_get_triangles(ctx, buf, count * 3)
    if rc != 0:
        raise RuntimeError(f"p2t_get_triangles failed, rc={rc}")
    _lib.p2t_destroy(ctx)
    tris = [tuple(buf[3*i:3*i+3]) for i in range(count)]
    return tris


# ---------------------------------------------------------------
# Mesh construction — contour trace + DP simplify (cv2, flagged
# non-identical to Dart's tracer) + landmark nudge (Dart algorithm,
# ported exactly) + Poisson sampling (Dart algorithm/params, flagged
# non-identical PRNG) + real poly2tri.
# ---------------------------------------------------------------

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
    """Exact port of the Dart TopMeshGenerator._nudgeInward algorithm:
    find nearest boundary segment, compute its TRUE perpendicular
    normal, test both directions for genuine inwardness, place the
    point exactly `distance` along that direction from the nearest
    point on that segment."""
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
        raise RuntimeError("Neither perpendicular direction tested inward — degenerate boundary geometry")
    return tuple(best_q + inward*distance)


def poisson_interior(boundary, extra_seeds, min_spacing, min_clearance, seed, max_attempts, w, h):
    """Same algorithm/parameters as Dart's poissonSegmentAware.
    PRNG is numpy's Mersenne Twister, NOT Dart's Random — same seed
    value does not produce the same point sequence. Flagged, not
    hidden."""
    rng = np.random.RandomState(seed)
    accepted = list(boundary) + list(extra_seeds)
    out = []
    poly = boundary
    for _ in range(max_attempts):
        c = (rng.uniform(0, w), rng.uniform(0, h))
        if not point_in_polygon(c, poly):
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
    print(f"  Boundary: {n_boundary} vertices (Dart's own tracer produced 44 on this silhouette — see flagged mismatch)")

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
            print(f"  Nudged {name}: {pt} -> {final_pt} (clearance={STEINER_MIN_CLEARANCE}px)")
        steiner_pts.append(final_pt)
        landmark_index[name] = idx
        idx += 1

    poisson_pts = poisson_interior(boundary, steiner_pts, POISSON_MIN_SPACING, POISSON_MIN_CLEARANCE,
                                    POISSON_SEED, POISSON_MAX_ATTEMPTS, w, h)
    print(f"  Poisson interior: {len(poisson_pts)} points")

    all_vertices = boundary + steiner_pts + poisson_pts
    triangles = real_poly2tri_triangulate(boundary, steiner_pts + poisson_pts)
    print(f"  Triangulated (real poly2tri engine): {len(triangles)} triangles")

    return np.array(all_vertices), triangles, landmark_index, n_boundary


# ---------------------------------------------------------------
# Geodesic (mesh-graph) distance — the ONE variable changed from E.
# Dijkstra from a segment's two endpoint vertices simultaneously.
# ---------------------------------------------------------------

def build_adjacency(vertices, triangles):
    adj = {i: [] for i in range(len(vertices))}
    seen_edges = set()
    for tri in triangles:
        for a, b in [(tri[0],tri[1]), (tri[1],tri[2]), (tri[2],tri[0])]:
            key = tuple(sorted((a, b)))
            if key in seen_edges:
                continue
            seen_edges.add(key)
            w = float(np.hypot(*(vertices[a] - vertices[b])))
            adj[a].append((b, w))
            adj[b].append((a, w))
    return adj


def multi_source_dijkstra(adj, sources, n):
    dist = [float('inf')] * n
    pq = []
    for s in sources:
        dist[s] = 0.0
        heapq.heappush(pq, (0.0, s))
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


# ---------------------------------------------------------------
# C-minimal blend — IDENTICAL to E's formula, only dist_to_segment
# replaced with precomputed graph distance.
# ---------------------------------------------------------------

def blend_geodesic(idx, ua_landmark_indices, sh_landmark_indices, geo_dist_ua, geo_dist_sh, T_ua, T_sh, vertices):
    p = vertices[idx]
    if idx in ua_landmark_indices:
        return (np.array(T_ua) @ np.array([p[0], p[1], 1.0]))[:2], 1.0, 0.0
    if idx in sh_landmark_indices:
        return (np.array(T_sh) @ np.array([p[0], p[1], 1.0]))[:2], 0.0, 1.0
    d_ua, d_sh = geo_dist_ua[idx], geo_dist_sh[idx]
    w_ua_raw = 1.0 / (d_ua ** POWER + EPSILON)
    w_sh_raw = 1.0 / (d_sh ** POWER + EPSILON)
    total = w_ua_raw + w_sh_raw
    w_ua, w_sh = w_ua_raw/total, w_sh_raw/total
    pred_ua = (np.array(T_ua) @ np.array([p[0], p[1], 1.0]))[:2]
    pred_sh = (np.array(T_sh) @ np.array([p[0], p[1], 1.0]))[:2]
    return w_ua*pred_ua + w_sh*pred_sh, w_ua, w_sh


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

    print("\nBuilding dense mesh (real poly2tri, flagged tracer/PRNG mismatches)...")
    vertices, triangles, landmark_index, n_boundary = build_dense_mesh(garment)
    print(f"\n{'='*70}\nDIAGNOSTIC — landmark_index contents\n{'='*70}")
    for name in ["left_underarm", "right_underarm", "left_shoulder", "right_shoulder"]:
        i = landmark_index[name]
        raw = {"left_underarm": garment.left_underarm, "right_underarm": garment.right_underarm,
               "left_shoulder": garment.left_shoulder, "right_shoulder": garment.right_shoulder}[name]
        print(f"  {name:16s} index={i}  vertices[index]={tuple(vertices[i])}  raw_landmark={raw}")
    print(f"  ua_idx set = {ua_idx if 'ua_idx' in dir() else 'not yet built'}")

    g_ua = (garment.left_underarm, garment.right_underarm) if garment.left_underarm[0] < garment.right_underarm[0] else (garment.right_underarm, garment.left_underarm)
    g_sh = (garment.left_shoulder, garment.right_shoulder) if garment.left_shoulder[0] < garment.right_shoulder[0] else (garment.right_shoulder, garment.left_shoulder)
    a_ua = (avatar.left_underarm, avatar.right_underarm) if avatar.left_underarm[0] < avatar.right_underarm[0] else (avatar.right_underarm, avatar.left_underarm)
    a_sh = (avatar.left_shoulder, avatar.right_shoulder) if avatar.left_shoulder[0] < avatar.right_shoulder[0] else (avatar.right_shoulder, avatar.left_shoulder)

    # FROZEN from E — identical call.
    T_ua = solve_similarity_2point(list(g_ua), list(a_ua))
    T_sh = solve_similarity_2point(list(g_sh), list(a_sh))
    print(f"\nT_underarm (frozen, identical to E): {np.array(T_ua).tolist()}")
    print(f"T_shoulder (frozen, identical to E): {np.array(T_sh).tolist()}")

    ua_idx = {landmark_index["left_underarm"], landmark_index["right_underarm"]}
    sh_idx = {landmark_index["left_shoulder"], landmark_index["right_shoulder"]}
    
    print(f"  ua_idx = {ua_idx}   sh_idx = {sh_idx}")
    print(f"  left_shoulder index {landmark_index['left_shoulder']} in sh_idx: {landmark_index['left_shoulder'] in sh_idx}")
    print(f"  right_shoulder index {landmark_index['right_shoulder']} in sh_idx: {landmark_index['right_shoulder'] in sh_idx}")

    print("\nComputing graph (geodesic) distances via Dijkstra...")
    adj = build_adjacency(vertices, triangles)
    geo_dist_ua = multi_source_dijkstra(adj, list(ua_idx), len(vertices))
    geo_dist_sh = multi_source_dijkstra(adj, list(sh_idx), len(vertices))

    deformed = np.array([blend_geodesic(i, ua_idx, sh_idx, geo_dist_ua, geo_dist_sh, T_ua, T_sh, vertices)[0]
                          for i in range(len(vertices))])

    # 1-2. Landmark error
    print(f"\n{'='*70}\n1-2. LANDMARK POSITIONAL ERROR\n{'='*70}")
    for name in ["left_underarm", "right_underarm", "left_shoulder", "right_shoulder"]:
        i = landmark_index[name]
        target = {"left_underarm": avatar.left_underarm, "right_underarm": avatar.right_underarm,
                  "left_shoulder": avatar.left_shoulder, "right_shoulder": avatar.right_shoulder}[name]
        from_deformed_array = deformed[i]
        fresh_call, w_ua_fresh, w_sh_fresh = blend_geodesic(i, ua_idx, sh_idx, geo_dist_ua, geo_dist_sh, T_ua, T_sh, vertices)
        err_array = float(np.hypot(*(from_deformed_array - np.array(target))))
        err_fresh = float(np.hypot(*(fresh_call - np.array(target))))
        print(f"  {name:16s} index={i}")
        print(f"    deformed[{i}]        = {tuple(from_deformed_array)}  error={err_array:.4f}px")
        print(f"    fresh blend_geodesic = {tuple(fresh_call)}  (w_ua={w_ua_fresh}, w_sh={w_sh_fresh})  error={err_fresh:.4f}px")
        print(f"    target               = {target}")
        
    # 4-6. Structural integrity
    print(f"\n{'='*70}\n4-6. MESH STRUCTURAL INTEGRITY\n{'='*70}")
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

    # 7-8. Hem / sleeve behavior — identical test points to E, by position
    print(f"\n{'='*70}\n7-8. HEM AND SLEEVE BEHAVIOR\n{'='*70}")
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
    predicted = {}
    for name, pt in test_points.items():
        nearest = int(np.argmin(np.linalg.norm(vertices - np.array(pt), axis=1)))
        d, w_ua, w_sh = blend_geodesic(nearest, ua_idx, sh_idx, geo_dist_ua, geo_dist_sh, T_ua, T_sh, vertices)
        predicted[name] = d
        print(f"  {name:18s} nearest_vertex={nearest}  geo_dist_ua={geo_dist_ua[nearest]:.1f}  "
              f"geo_dist_sh={geo_dist_sh[nearest]:.1f}  w_ua={w_ua:.3f} w_sh={w_sh:.3f}  -> {tuple(d)}")

    # Render
    garment_arr = np.array(garment.image.convert("RGBA"))
    avatar_rgba = np.array(avatar.image.convert("RGBA"))
    aw, ah = avatar.image.size
    layer = render_mesh(vertices, deformed, triangles, garment_arr, aw, ah)
    composite = avatar_rgba.copy()
    alpha = layer[:,:,3:4].astype(float)/255.0
    composite[:,:,:3] = (avatar_rgba[:,:,:3]*(1-alpha) + layer[:,:,:3]*alpha).astype(np.uint8)
    Image.fromarray(composite).convert("RGB").save(f"{OUTPUT_DIR}/checkpoint4_C_render.png")

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
    annotated.save(f"{OUTPUT_DIR}/checkpoint4_C_render_annotated.png")

    print(f"\nSaved: {OUTPUT_DIR}/checkpoint4_C_render.png")
    print(f"Saved: {OUTPUT_DIR}/checkpoint4_C_render_annotated.png")
    print(f"\n>>> Compare directly against checkpoint4_E_render_annotated.png — same garment, same avatar,")
    print(f">>> same 5 points, same T_underarm/T_shoulder, only the distance metric changed.")


if __name__ == "__main__":
    main()