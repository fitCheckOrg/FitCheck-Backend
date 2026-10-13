"""
tryon/checkpoint5_ARAP0_head_investigation.py

Where did the missing head go, in coordinate-space terms — BEFORE
the compositor runs. Same deterministic mesh/solve as the validation
script. Reports the deformed garment's actual bounding box against
the avatar canvas and the avatar's own head-region pixels, to
distinguish:

  A. Geometry  — the deformed garment genuinely extends up over the
                 head region in its own coordinates.
  B. Bounds    — the deformed garment extends beyond the avatar
                 canvas and gets clipped/reframed.
  C. Compositor — the garment stays within a sensible region, but
                 something in alpha/compositing masks the avatar.

Run: python -m tryon.checkpoint5_ARAP0_head_investigation
"""

import os
import ctypes
import numpy as np
import cv2
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from PIL import Image

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
MAX_ITERATIONS = 60

# ---------------------------------------------------------------
# Identical mesh construction + ARAP solver (see checkpoint5_ARAP0.py
# / checkpoint5_ARAP0_validation.py for the same code, reproduced here
# for a self-contained script).
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
    for it in range(max_iterations):
        rotations = local_step(rest, current, adj, n)
        current = global_step(L_factorized, rest, adj, rotations, n, handle_indices, handle_targets)
    return current


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

    adj = build_adjacency(n, triangles)
    print(f"Re-running deterministic ARAP-0 solve ({MAX_ITERATIONS} iterations)...")
    deformed = run_arap(vertices, adj, handle_indices, handle_targets, MAX_ITERATIONS)

    # ============================================================
    # THE ACTUAL QUESTION: coordinate-space comparison, pre-compositor
    # ============================================================
    aw, ah = avatar.image.size
    print(f"\n{'='*70}\nCOORDINATE-SPACE INVESTIGATION\n{'='*70}")
    print(f"Avatar canvas size: {aw} x {ah}")

    gx0, gy0 = deformed[:,0].min(), deformed[:,1].min()
    gx1, gy1 = deformed[:,0].max(), deformed[:,1].max()
    print(f"\nDeformed garment bounding box (pre-compositor, raw mesh coordinates):")
    print(f"  x: [{gx0:.2f}, {gx1:.2f}]   y: [{gy0:.2f}, {gy1:.2f}]")
    print(f"  width={gx1-gx0:.2f}  height={gy1-gy0:.2f}")

    print(f"\nTest A — does the garment extend ABOVE y=0 (off the top of the canvas)?")
    print(f"  garment top y = {gy0:.2f}   {'YES — extends above canvas top' if gy0 < 0 else 'NO — stays within canvas vertically'}")

    print(f"\nTest B — does the garment extend beyond the canvas width [0, {aw}]?")
    print(f"  garment x range: [{gx0:.2f}, {gx1:.2f}]")
    outside_left = gx0 < 0
    outside_right = gx1 > aw
    print(f"  {'YES, extends left of x=0' if outside_left else 'stays within left bound'}")
    print(f"  {'YES, extends right of x='+str(aw) if outside_right else 'stays within right bound'}")

    # Where is the avatar's own head, roughly, so we can check direct overlap
    # (shoulders are the lowest reliable landmark we have; head is above them)
    a_shoulder_y = min(avatar.left_shoulder[1], avatar.right_shoulder[1])
    print(f"\nAvatar shoulder y-coordinate (head region is ABOVE this): {a_shoulder_y:.2f}")
    print(f"Deformed garment top y: {gy0:.2f}")
    print(f"Garment top is {'ABOVE' if gy0 < a_shoulder_y else 'BELOW'} the avatar's own shoulder landmark by {abs(gy0 - a_shoulder_y):.2f}px")

    # Which specific vertices are responsible for the topmost extent
    top_vertex_idx = int(np.argmin(deformed[:,1]))
    print(f"\nTopmost deformed vertex: index {top_vertex_idx}")
    print(f"  rest position:     {tuple(vertices[top_vertex_idx])}")
    print(f"  deformed position: {tuple(deformed[top_vertex_idx])}")
    print(f"  is a handle: {top_vertex_idx in handle_indices}")

    # Save the deformed mesh overlaid on a BLANK canvas the exact size of the
    # avatar image, with the canvas boundary drawn explicitly — isolates
    # geometry from the compositor entirely.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    from PIL import ImageDraw
    canvas = Image.new("RGB", (aw, ah), (40, 40, 40))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, aw-1, ah-1], outline=(255, 0, 0), width=2)  # canvas bounds in red
    for tri in triangles:
        pts = [tuple(deformed[i]) for i in tri]
        draw.polygon(pts, outline=(0, 255, 0))
    for i in handle_indices:
        x, y = deformed[i]
        draw.ellipse([x-5, y-5, x+5, y+5], fill=(255, 255, 0))
    y_shoulder = a_shoulder_y
    draw.line([(0, y_shoulder), (aw, y_shoulder)], fill=(0, 200, 255), width=1)
    canvas.save(f"{OUTPUT_DIR}/checkpoint5_ARAP0_wireframe_vs_canvas.png")
    print(f"\nSaved: {OUTPUT_DIR}/checkpoint5_ARAP0_wireframe_vs_canvas.png")
    print(f"  Red = avatar canvas bounds. Green = deformed mesh wireframe.")
    print(f"  Yellow dots = the 4 handles. Cyan line = avatar's shoulder y-level (head is above this line).")


if __name__ == "__main__":
    main()