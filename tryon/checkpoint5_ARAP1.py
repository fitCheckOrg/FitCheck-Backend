"""
tryon/checkpoint5_ARAP1.py

ARAP-1 — Decision 1's predefined fallback, with Decision 6 = R1.

ONLY change from ARAP-0: each single-vertex hard handle h is replaced by
its graph 1-ring  R(h) = {h} ∪ N(h), and every ring vertex v is hard-
constrained to  t_v = p_v + (t_h - p_h)  (the ring translates as a block
by its handle's displacement; no scale or rotation information added).

Frozen (imported, not copied): mesh construction, adjacency, uniform
weights, cold-start init (constrained vertices snapped, rest at rest
position), local/global solver, 60-iteration run, overlap detector.

ARAP-0 is re-run in the same process as the baseline so every metric is
computed by identical code for both.

Run: python -m tryon.checkpoint5_ARAP1
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw

from tryon.avatar_representation import build_avatar_representation
from tryon.garment_representation import build_garment_representation
from tryon.checkpoint5_ARAP0_deformation_field import (
    build_dense_mesh, build_adjacency, run_arap, detect_overlap,
    GOLDEN_GARMENT_ID, AVATAR_USER_ID, MAX_ITERATIONS,
)

OUTPUT_DIR = "outputs"
BIG_JUMP_DEG = 45.0  # reporting threshold for counting only, not a quality criterion


def multi_bfs(adj, sources, n):
    dist = [10**9] * n
    q = []
    for s in sources:
        dist[s] = 0
        q.append(s)
    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        for v in adj[u]:
            if dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def get_anatomical_handles(garment, avatar, landmark_index):
    """Same coordinate-pairing convention as every ARAP-0 script."""
    a_ua = (avatar.left_underarm, avatar.right_underarm) if avatar.left_underarm[0] < avatar.right_underarm[0] else (avatar.right_underarm, avatar.left_underarm)
    a_sh = (avatar.left_shoulder, avatar.right_shoulder) if avatar.left_shoulder[0] < avatar.right_shoulder[0] else (avatar.right_shoulder, avatar.left_shoulder)
    left_ua_first = garment.left_underarm[0] < garment.right_underarm[0]
    left_sh_first = garment.left_shoulder[0] < garment.right_shoulder[0]
    pairs = [
        ("left_underarm" if left_ua_first else "right_underarm", a_ua[0]),
        ("right_underarm" if left_ua_first else "left_underarm", a_ua[1]),
        ("left_shoulder" if left_sh_first else "right_shoulder", a_sh[0]),
        ("right_shoulder" if left_sh_first else "left_shoulder", a_sh[1]),
    ]
    targets = {}
    for name, a_pt in pairs:
        targets[landmark_index[name]] = np.array(a_pt, dtype=float)
    return targets


def build_rings(adj, handle_targets):
    rings = {h: {h} | set(adj[h]) for h in handle_targets}
    problems = []
    handles = set(handle_targets)
    for h, ring in rings.items():
        # verify it really is a 1-ring of h
        if not all((v == h) or (v in adj[h]) for v in ring) or not set(adj[h]) <= ring:
            problems.append(f"ring of {h} is not exactly {{h}} ∪ N(h)")
        others = (handles - {h}) & ring
        if others:
            problems.append(f"ring of {h} contains other anatomical handle(s) {sorted(others)}")
    hs = list(rings)
    for a in range(len(hs)):
        for b in range(a + 1, len(hs)):
            inter = rings[hs[a]] & rings[hs[b]]
            if inter:
                problems.append(f"rings of {hs[a]} and {hs[b]} overlap at vertices {sorted(inter)}")
    return rings, problems


def ring_targets(rest, rings, handle_targets):
    targets = {}
    for h, ring in rings.items():
        disp = handle_targets[h] - rest[h]
        for v in ring:
            targets[v] = rest[v] + disp
    return targets


def angle_of(R):
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def analyze(rest, deformed, rotations, triangles, adj, constrained, anat_handles,
            constraint_targets, canvas_w, shoulder_y):
    n = len(rest)
    out = {}
    out["residual"] = max(float(np.linalg.norm(deformed[v] - t)) for v, t in constraint_targets.items())
    hops_c = multi_bfs(adj, constrained, n)   # hops to this run's own constrained set
    hops_h = multi_bfs(adj, anat_handles, n)  # hops to the 4 anatomical handle vertices (common axis)

    def area2(P, tri):
        a, b, c = P[list(tri)]
        return (b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])

    inverted, degenerate = [], 0
    for ti, tri in enumerate(triangles):
        s_def = area2(deformed, tri)
        if abs(s_def) < 1e-6:
            degenerate += 1
            continue
        if area2(rest, tri) * s_def < 0:
            inverted.append(ti)
    out["inverted"], out["degenerate"] = inverted, degenerate
    out["inv_hops_c"] = [min(hops_c[v] for v in triangles[t]) for t in inverted]
    out["inv_hops_h"] = [min(hops_h[v] for v in triangles[t]) for t in inverted]

    T = len(triangles)
    tri_pts = [deformed[list(t)] for t in triangles]
    tri_bbox = [(p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()) for p in tri_pts]
    tri_sets = [set(t) for t in triangles]
    overlaps = []
    for i in range(T):
        for j in range(i + 1, T):
            if tri_sets[i] & tri_sets[j]:
                continue
            bi, bj = tri_bbox[i], tri_bbox[j]
            if bi[1] < bj[0] or bj[1] < bi[0] or bi[3] < bj[2] or bj[3] < bi[2]:
                continue
            area = detect_overlap(*tri_pts[i], *tri_pts[j])
            if area > 0.5:
                verts = tri_sets[i] | tri_sets[j]
                overlaps.append((i, j, area, min(hops_c[v] for v in verts), min(hops_h[v] for v in verts)))
    out["overlaps"] = overlaps

    rot_edges = []
    ratios, band_ratios = [], {0: [], 1: [], 2: [], 3: []}
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            diff = abs(((angle_of(rotations[i]) - angle_of(rotations[j])) + 180) % 360 - 180)
            rot_edges.append((i, j, diff, min(hops_c[i], hops_c[j]), min(hops_h[i], hops_h[j])))
            rest_len = float(np.linalg.norm(rest[i] - rest[j]))
            if rest_len > 1e-9:
                r = float(np.linalg.norm(deformed[i] - deformed[j])) / rest_len
                ratios.append(r)
                band_ratios[min(min(hops_c[i], hops_c[j]), 3)].append(r)
    out["rot_edges"], out["ratios"], out["band_ratios"] = rot_edges, np.array(ratios), band_ratios

    rest_w = rest[:, 0].max() - rest[:, 0].min()
    rest_h = rest[:, 1].max() - rest[:, 1].min()
    def_w = deformed[:, 0].max() - deformed[:, 0].min()
    def_h = deformed[:, 1].max() - deformed[:, 1].min()
    out["bbox"] = dict(w_ratio=def_w / rest_w, h_ratio=def_h / rest_h,
                       top_y=float(deformed[:, 1].min()), max_x=float(deformed[:, 0].max()),
                       above_shoulder=float(shoulder_y - deformed[:, 1].min()),
                       right_overflow=float(max(0.0, deformed[:, 0].max() - canvas_w)))
    return out


def hop_hist(values):
    h = [0, 0, 0, 0]
    for x in values:
        h[min(x, 3)] += 1
    return h


def render_mesh(vertices_rest, vertices_deformed, triangles, texture_arr, canvas_w, canvas_h):
    out = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    tex_h, tex_w = texture_arr.shape[:2]
    for tri in triangles:
        rest, dst = vertices_rest[list(tri)], vertices_deformed[list(tri)]
        x0, y0 = np.floor(dst.min(axis=0)).astype(int)
        x1, y1 = np.ceil(dst.max(axis=0)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, canvas_w), min(y1, canvas_h)
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

    print("Rebuilding real AvatarRepresentation and GarmentRepresentation...")
    avatar = build_avatar_representation()
    garment = build_garment_representation(GOLDEN_GARMENT_ID, AVATAR_USER_ID)

    rest, triangles, landmark_index, n_boundary = build_dense_mesh(garment)
    n = len(rest)
    adj = build_adjacency(n, triangles)
    handle_targets = get_anatomical_handles(garment, avatar, landmark_index)
    anat_handles = set(handle_targets.keys())
    canvas_w, canvas_h = avatar.image.size
    shoulder_y = float(min(avatar.left_shoulder[1], avatar.right_shoulder[1]))
    print(f"Mesh: {n} vertices, {len(triangles)} triangles. Anatomical handles: {sorted(anat_handles)}")

    # ---------------- ring construction + validation ----------------
    print(f"\n{'='*70}\nRING CONSTRUCTION AND VALIDATION\n{'='*70}")
    rings, problems = build_rings(adj, handle_targets)
    for h, ring in rings.items():
        print(f"  handle {h}: ring size {len(ring)}  members={sorted(ring)}")
    hs = list(rings)
    for a in range(len(hs)):
        for b in range(a + 1, len(hs)):
            d = multi_bfs(adj, rings[hs[a]], n)
            print(f"  min graph distance, ring({hs[a]}) -> ring({hs[b]}): {min(d[v] for v in rings[hs[b]])} hops")
    if problems:
        print("\nRING VALIDATION FAILED — not resolving silently. Problems:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("  All rings valid: exact 1-rings, no other handle inside any ring, mutually disjoint.")
    ring_set = set().union(*rings.values())
    ring_tgts = ring_targets(rest, rings, handle_targets)
    print(f"  Total constrained vertices in ARAP-1: {len(ring_set)} (ARAP-0: {len(anat_handles)})")

    # ---------------- solves ----------------
    print(f"\nRunning ARAP-0 baseline ({MAX_ITERATIONS} iterations)...")
    def0, rot0 = run_arap(rest, adj, anat_handles, handle_targets, MAX_ITERATIONS)
    print(f"Running ARAP-1 (R1 rings, {MAX_ITERATIONS} iterations)...")
    def1, rot1 = run_arap(rest, adj, ring_set, ring_tgts, MAX_ITERATIONS)

    a0 = analyze(rest, def0, rot0, triangles, adj, anat_handles, anat_handles, handle_targets, canvas_w, shoulder_y)
    a1 = analyze(rest, def1, rot1, triangles, adj, ring_set, anat_handles, ring_tgts, canvas_w, shoulder_y)

    def both(label, f0, f1):
        print(f"  {label:<44} ARAP-0: {f0:<16} ARAP-1: {f1}")

    # ---------------- headline comparison ----------------
    print(f"\n{'='*70}\nHEADLINE COMPARISON (identical code path for both)\n{'='*70}")
    both("max constraint residual (px)", f"{a0['residual']:.6f}", f"{a1['residual']:.6f}")
    both("inverted triangles (vs own rest orientation)", f"{len(a0['inverted'])}/{len(triangles)}", f"{len(a1['inverted'])}/{len(triangles)}")
    both("degenerate triangles", a0['degenerate'], a1['degenerate'])
    both("precise overlap pairs", len(a0['overlaps']), len(a1['overlaps']))
    both("max overlap area (px²)", f"{max([o[2] for o in a0['overlaps']], default=0):.1f}", f"{max([o[2] for o in a1['overlaps']], default=0):.1f}")
    both("total overlap area (px²)", f"{sum(o[2] for o in a0['overlaps']):.1f}", f"{sum(o[2] for o in a1['overlaps']):.1f}")
    print(f"  (baseline cross-check: earlier scripts reported 39 overlap pairs and 5 inverted triangles for ARAP-0)")

    # ---------------- rotation field ----------------
    print(f"\n{'='*70}\nROTATION FIELD (neighbor angle difference, degrees)\n{'='*70}")
    for label, a in (("ARAP-0", a0), ("ARAP-1", a1)):
        d = np.array([e[2] for e in a["rot_edges"]])
        print(f"  {label}: min={d.min():.2f} max={d.max():.2f} mean={d.mean():.2f} median={np.median(d):.2f}  "
              f"edges>{BIG_JUMP_DEG:.0f}°: {int((d > BIG_JUMP_DEG).sum())}  edges>90°: {int((d > 90).sum())}")
    for label, a in (("ARAP-0", a0), ("ARAP-1", a1)):
        top = sorted(a["rot_edges"], key=lambda e: -e[2])[:5]
        print(f"  {label} five largest jumps (i, j, Δ°, hops_to_own_constraints, hops_to_anatomical_handles):")
        for i, j, diff, hc, hh in top:
            print(f"    ({i:3d},{j:3d})  {diff:6.2f}°  hc={hc}  hh={hh}")

    # ---------------- edge-length field ----------------
    print(f"\n{'='*70}\nGLOBAL EDGE-LENGTH RATIO (deformed/rest) — percentiles over all edges\n{'='*70}")
    pct = [0, 5, 25, 50, 75, 95, 100]
    print(f"  percentile        " + "  ".join(f"{p:>6}" for p in pct))
    for label, a in (("ARAP-0", a0), ("ARAP-1", a1)):
        print(f"  {label}            " + "  ".join(f"{v:6.3f}" for v in np.percentile(a["ratios"], pct)) + f"   mean={a['ratios'].mean():.3f}")
    print(f"\n  Mean / max ratio by hop band from each run's OWN constrained set (band 3 = 3+ hops):")
    for label, a in (("ARAP-0", a0), ("ARAP-1", a1)):
        cells = []
        for b in range(4):
            r = a["band_ratios"][b]
            cells.append(f"band{b}: n={len(r):3d} mean={np.mean(r) if r else float('nan'):.3f} max={max(r) if r else float('nan'):.3f}")
        print(f"  {label}: " + " | ".join(cells))

    # ---------------- bounding box / head ----------------
    print(f"\n{'='*70}\nBOUNDING BOX AND HEAD-REGION CHECK\n{'='*70}")
    both("deformed/rest width ratio", f"{a0['bbox']['w_ratio']:.3f}", f"{a1['bbox']['w_ratio']:.3f}")
    both("deformed/rest height ratio", f"{a0['bbox']['h_ratio']:.3f}", f"{a1['bbox']['h_ratio']:.3f}")
    both("garment top y (canvas px)", f"{a0['bbox']['top_y']:.1f}", f"{a1['bbox']['top_y']:.1f}")
    both("garment top above avatar shoulder line (px)", f"{a0['bbox']['above_shoulder']:.1f}", f"{a1['bbox']['above_shoulder']:.1f}")
    both(f"overflow past canvas right edge x={canvas_w} (px)", f"{a0['bbox']['right_overflow']:.1f}", f"{a1['bbox']['right_overflow']:.1f}")

    # ---------------- where the pathology lives ----------------
    print(f"\n{'='*70}\nWHERE THE PATHOLOGY LIVES — counts by hop distance (0, 1, 2, 3+)\n{'='*70}")
    print(f"  (hc = hops to the run's OWN constrained set; hh = hops to the 4 anatomical handle vertices — common axis)")
    for label, a in (("ARAP-0", a0), ("ARAP-1", a1)):
        print(f"  {label}")
        print(f"    overlap pairs      hc={hop_hist([o[3] for o in a['overlaps']])}  hh={hop_hist([o[4] for o in a['overlaps']])}")
        print(f"    inverted triangles hc={hop_hist(a['inv_hops_c'])}  hh={hop_hist(a['inv_hops_h'])}")
        big = [e for e in a["rot_edges"] if e[2] > BIG_JUMP_DEG]
        print(f"    rot jumps >{BIG_JUMP_DEG:.0f}°     hc={hop_hist([e[3] for e in big])}  hh={hop_hist([e[4] for e in big])}")

    # ---------------- images for ARAP-1 ----------------
    print("\nRendering ARAP-1 composite + wireframe...")
    garment_arr = np.array(garment.image.convert("RGBA"))
    avatar_rgba = np.array(avatar.image.convert("RGBA"))
    layer = render_mesh(rest, def1, triangles, garment_arr, canvas_w, canvas_h)
    composite = avatar_rgba.copy()
    alpha = layer[:, :, 3:4].astype(float) / 255.0
    composite[:, :, :3] = (avatar_rgba[:, :, :3] * (1 - alpha) + layer[:, :, :3] * alpha).astype(np.uint8)
    annotated = Image.fromarray(composite).convert("RGB")
    draw = ImageDraw.Draw(annotated)

    mask = garment.mask
    ys, xs = np.where(mask)
    bottom_y = ys.max()
    bottom_xs = xs[ys == bottom_y]
    test_points = {
        "left_hem": ((float(bottom_xs.min()), float(bottom_y)), (255, 0, 0)),
        "center_hem": ((float(np.median(bottom_xs)), float(bottom_y)), (255, 128, 0)),
        "right_hem": ((float(bottom_xs.max()), float(bottom_y)), (255, 0, 255)),
        "left_sleeve_end": ((float(xs.min()), float(ys[xs == xs.min()].mean())), (0, 255, 255)),
        "right_sleeve_end": ((float(xs.max()), float(ys[xs == xs.max()].mean())), (0, 255, 0)),
    }
    print("  Diagnostic points (nearest vertex): rest -> ARAP-0 -> ARAP-1")
    for name, (pt, color) in test_points.items():
        nearest = int(np.argmin(np.linalg.norm(rest - np.array(pt), axis=1)))
        x, y = def1[nearest]
        draw.ellipse([x-6, y-6, x+6, y+6], outline=color, width=3)
        print(f"    {name:18s} v{nearest}: {tuple(np.round(rest[nearest],1))} -> {tuple(np.round(def0[nearest],1))} -> {tuple(np.round(def1[nearest],1))}")
    for t in handle_targets.values():
        draw.ellipse([t[0]-5, t[1]-5, t[0]+5, t[1]+5], outline=(255, 255, 0), width=2)
    annotated.save(f"{OUTPUT_DIR}/checkpoint5_ARAP1_render_annotated.png")

    canvas = Image.new("RGB", (canvas_w, canvas_h), (40, 40, 40))
    cd = ImageDraw.Draw(canvas)
    cd.rectangle([0, 0, canvas_w - 1, canvas_h - 1], outline=(255, 0, 0), width=2)
    for tri in triangles:
        cd.polygon([tuple(def1[i]) for i in tri], outline=(0, 255, 0))
    for v in ring_set:
        x, y = def1[v]
        cd.ellipse([x-3, y-3, x+3, y+3], fill=(255, 140, 0))
    for h in anat_handles:
        x, y = def1[h]
        cd.ellipse([x-5, y-5, x+5, y+5], fill=(255, 255, 0))
    cd.line([(0, shoulder_y), (canvas_w, shoulder_y)], fill=(0, 200, 255), width=1)
    canvas.save(f"{OUTPUT_DIR}/checkpoint5_ARAP1_wireframe_vs_canvas.png")
    print(f"\nSaved: {OUTPUT_DIR}/checkpoint5_ARAP1_render_annotated.png")
    print(f"Saved: {OUTPUT_DIR}/checkpoint5_ARAP1_wireframe_vs_canvas.png")
    print("  Wireframe: red = canvas bounds, green = mesh, orange = ring vertices, yellow = anatomical handles, cyan = avatar shoulder line.")


if __name__ == "__main__":
    main()