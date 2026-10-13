"""
tryon/checkpoint5_ARAP1_R2.py

ARAP-1 R2 — similarity-transformed 1-ring constraints.

R1 (locked earlier): ring vertex v of handle h is constrained to
    t_v = p_v + (t_h - p_h)          (translation only)
R2 (this script): ring vertex v of handle h is constrained to
    t_v = T_h(p_v)                    (the handle's PAIR similarity transform:
                                       T_underarm for both underarm rings,
                                       T_shoulder for both shoulder rings —
                                       the same solve_similarity_2point matrices E used)

ONE deliberate detail: the handle vertex ITSELF keeps its exact avatar-landmark
target (as in ARAP-0 and R1). T_h(p_h) differs from that target by the ~0.4px
mesh-vertex-vs-raw-landmark offset already known on this mesh; this script
prints that offset per handle rather than hiding it.

Everything else is frozen by IMPORT, not copy: mesh, adjacency, uniform weights,
cold-start init (constrained vertices snapped, rest at rest position), solver,
iteration cap, overlap detector, ring construction, and the ARAP-1 analysis.
ARAP-0 and R1 are re-run in the same process so all three columns come from
identical code.

ADDITIONS (read-only; applied identically to all three runs):
  - locations of the 5 most compressed / 5 most stretched edges
  - overlap pairs that touch two different rings (the "ring crowding" check)
  - ring-internal edge ratios (verifies R2 was built as specified)
  - min deformed distance between different rings
  - far-field edge ratio defined by hops from the anatomical handles (all runs)

Run: python -m tryon.checkpoint5_ARAP1_R2
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw

from tryon.avatar_representation import build_avatar_representation
from tryon.garment_representation import build_garment_representation
from tryon.piecewise_continuous_test import solve_similarity_2point
from tryon.checkpoint5_ARAP0_deformation_field import (
    build_dense_mesh, build_adjacency, run_arap,
    GOLDEN_GARMENT_ID, AVATAR_USER_ID, MAX_ITERATIONS,
)
from tryon.checkpoint5_ARAP1 import (
    get_anatomical_handles, build_rings, ring_targets, analyze,
    hop_hist, multi_bfs, render_mesh, BIG_JUMP_DEG,
)

OUTPUT_DIR = "outputs"

REGISTERED = """
======================================================================
REGISTERED BEFORE RUNNING (R2)
======================================================================
Experiment claim: R2 tests whether a locally similarity-consistent constraint
region resolves the deformation pathology WITHOUT changing ARAP's internal energy.

Near-field predictions (R2 vs R1, both computed in this run):
  N1  min edge ratio is higher than R1's
  N2  5th-percentile edge ratio is higher than R1's
  N3  max neighbor rotation jump is lower than R1's
  N4  number of neighbor rotation jumps > 45 deg is lower than R1's
  N5  [conditional on ring crowding being materially responsible] overlap pairs lower than R1's
  N6  [conditional on ring crowding being materially responsible] inverted triangles lower than R1's

Far-field rule (the critical test: does R2 produce a GLOBAL CONTRACTION?):
  MID = (1 + max(pair scale)) / 2   (halfway between rest scale and the weaker required scale)
  Quantities (all three computed for ARAP-0, R1 and R2):
    robust width ratio  = (P95-P5 spread of deformed vertex x) / (same spread at rest)
    robust height ratio = (P95-P5 spread of deformed vertex y) / (same spread at rest)
    far-field mean edge ratio (edges with both endpoints >= 3 hops from every anatomical handle)
    all three > MID  -> NO global contraction  (the ARAP energy is the bottleneck, not the handles)
    all three < MID  -> global contraction     (handle-side constraint was the issue)
    otherwise        -> MIXED
  NOTE (revised before any R2 data existed): the first draft of this rule used the plain
  bounding-box width/height ratios. In ARAP-0 and R1 the bbox width ratio was 0.725, only 0.02
  above MID, and a bbox is set by single extreme vertices, so it could flip the verdict on
  its own. Robust spreads replace it in the rule. Bbox ratios are still printed, not used.
MID is a fixed descriptive line, not a quality threshold.
"""


def pair_by_x(a, b):
    return (a, b) if a[0] < b[0] else (b, a)


def robust_ratio(rest, deformed, axis):
    r = np.percentile(rest[:, axis], 95) - np.percentile(rest[:, axis], 5)
    d = np.percentile(deformed[:, axis], 95) - np.percentile(deformed[:, axis], 5)
    return float(d / r)


def apply_T(T, p):
    return (T @ np.array([float(p[0]), float(p[1]), 1.0]))[:2]


def extras(rest, deformed, triangles, adj, anat_handles, ring_set, ring_of, rings, overlaps, hh):
    n = len(rest)

    def lab(v):
        return "H" if v in anat_handles else ("R" if v in ring_set else "O")

    edges, far = [], []
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            rl = float(np.linalg.norm(rest[i] - rest[j]))
            if rl < 1e-9:
                continue
            r = float(np.linalg.norm(deformed[i] - deformed[j])) / rl
            edges.append((r, i, j))
            if min(hh[i], hh[j]) >= 3:
                far.append(r)
    edges.sort()
    out = {"far_mean": float(np.mean(far)) if far else float("nan"), "far_n": len(far),
           "smallest": edges[:5], "largest": edges[-5:][::-1], "lab": lab}

    cross = 0
    for (i, j, area, hc, hhh) in overlaps:
        touched = {ring_of[v] for v in (set(triangles[i]) | set(triangles[j])) if v in ring_of}
        if len(touched) >= 2:
            cross += 1
    out["cross_ring_overlaps"] = cross

    internal = {}
    for h, ring in rings.items():
        rs = []
        for a in ring:
            for b in adj[a]:
                if b in ring and b > a:
                    rl = float(np.linalg.norm(rest[a] - rest[b]))
                    if rl > 1e-9:
                        rs.append(float(np.linalg.norm(deformed[a] - deformed[b])) / rl)
        internal[h] = rs
    out["internal"] = internal

    hs = sorted(rings)
    dmin = {}
    for ia in range(len(hs)):
        for ib in range(ia + 1, len(hs)):
            a, b = hs[ia], hs[ib]
            dmin[(a, b)] = min(float(np.linalg.norm(deformed[u] - deformed[v]))
                               for u in rings[a] for v in rings[b])
    out["ring_min_dist"] = dmin
    return out


def draw_wire(deformed, triangles, ring_set, anat_handles, canvas_w, canvas_h, shoulder_y, title):
    img = Image.new("RGB", (canvas_w, canvas_h), (40, 40, 40))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, canvas_w - 1, canvas_h - 1], outline=(255, 0, 0), width=2)
    for tri in triangles:
        d.polygon([tuple(deformed[i]) for i in tri], outline=(0, 255, 0))
    for v in ring_set:
        x, y = deformed[v]
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 140, 0))
    for h in anat_handles:
        x, y = deformed[h]
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 255, 0))
    d.line([(0, shoulder_y), (canvas_w, shoulder_y)], fill=(0, 200, 255), width=1)
    d.text((6, 6), title, fill=(255, 255, 255))
    return img


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(REGISTERED)

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

    # ---------- pair similarity transforms (same construction as experiment E) ----------
    g_ua = pair_by_x(garment.left_underarm, garment.right_underarm)
    g_sh = pair_by_x(garment.left_shoulder, garment.right_shoulder)
    a_ua = pair_by_x(avatar.left_underarm, avatar.right_underarm)
    a_sh = pair_by_x(avatar.left_shoulder, avatar.right_shoulder)
    T_ua = np.array(solve_similarity_2point(list(g_ua), list(a_ua)))
    T_sh = np.array(solve_similarity_2point(list(g_sh), list(a_sh)))
    scale_ua = float(np.hypot(T_ua[0, 0], T_ua[1, 0]))
    scale_sh = float(np.hypot(T_sh[0, 0], T_sh[1, 0]))
    rot_ua = float(np.degrees(np.arctan2(T_ua[1, 0], T_ua[0, 0])))
    rot_sh = float(np.degrees(np.arctan2(T_sh[1, 0], T_sh[0, 0])))
    MID = (1.0 + max(scale_ua, scale_sh)) / 2.0
    print(f"\nT_underarm: scale={scale_ua:.4f} rotation={rot_ua:.2f} deg   "
          f"T_shoulder: scale={scale_sh:.4f} rotation={rot_sh:.2f} deg")
    print(f"MID line for the global-contraction rule = {MID:.4f}")

    names = ("left_underarm", "right_underarm", "left_shoulder", "right_shoulder")
    name_of, T_of = {}, {}
    for nm in names:
        v = landmark_index[nm]
        name_of[v] = nm
        T_of[v] = T_ua if "underarm" in nm else T_sh

    print(f"\n{'='*70}\nHANDLE / TRANSFORM CONSISTENCY\n{'='*70}")
    print("  (T applied to the RAW landmark should hit the avatar target exactly;")
    print("   T applied to the MESH VERTEX shows the known mesh-vs-raw offset)")
    for h in sorted(anat_handles):
        raw = np.array(getattr(garment, name_of[h]), dtype=float)
        tgt = handle_targets[h]
        e_raw = float(np.linalg.norm(apply_T(T_of[h], raw) - tgt))
        e_vtx = float(np.linalg.norm(apply_T(T_of[h], rest[h]) - tgt))
        print(f"  handle {h:3d} ({name_of[h]:14s}) raw={tuple(raw)} mesh_vertex={tuple(np.round(rest[h],2))}"
              f"  |T(raw)-target|={e_raw:.4f}px  |T(vertex)-target|={e_vtx:.4f}px")

    # ---------- rings ----------
    rings, problems = build_rings(adj, handle_targets)
    if problems:
        print("\nRING VALIDATION FAILED — not resolving silently:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    ring_set = set().union(*rings.values())
    ring_of = {v: h for h, ring in rings.items() for v in ring}
    print(f"\nRings valid (same construction as R1). Sizes: "
          + ", ".join(f"{h}:{len(r)}" for h, r in sorted(rings.items()))
          + f"  total constrained = {len(ring_set)}")

    r1_targets = ring_targets(rest, rings, handle_targets)
    r2_targets = {}
    for h, ring in rings.items():
        for v in ring:
            r2_targets[v] = handle_targets[h].copy() if v == h else apply_T(T_of[h], rest[v])

    print(f"\n{'='*70}\nR2 TARGET CONSTRUCTION CHECK (ring-internal edges should scale by the pair scale)\n{'='*70}")
    for h, ring in sorted(rings.items()):
        ratios = []
        for a in ring:
            for b in adj[a]:
                if b in ring and b > a:
                    rl = float(np.linalg.norm(rest[a] - rest[b]))
                    tl = float(np.linalg.norm(r2_targets[a] - r2_targets[b]))
                    if rl > 1e-9:
                        ratios.append(tl / rl)
        expect = scale_ua if "underarm" in name_of[h] else scale_sh
        print(f"  ring {h:3d} ({name_of[h]:14s}) expected scale {expect:.3f}   "
              f"target edge ratios: min={min(ratios):.3f} max={max(ratios):.3f}   "
              f"(deviation from expected comes from the handle keeping its exact landmark target)")

    # ---------- solves ----------
    print(f"\nRunning ARAP-0 ({MAX_ITERATIONS} it)...")
    def0, rot0 = run_arap(rest, adj, anat_handles, handle_targets, MAX_ITERATIONS)
    print("Running R1...")
    def1, rot1 = run_arap(rest, adj, ring_set, r1_targets, MAX_ITERATIONS)
    print("Running R2...")
    def2, rot2 = run_arap(rest, adj, ring_set, r2_targets, MAX_ITERATIONS)

    L0, L1, L2 = "ARAP-0", "R1", "R2"
    runs = [(L0, def0, rot0, anat_handles, handle_targets),
            (L1, def1, rot1, ring_set, r1_targets),
            (L2, def2, rot2, ring_set, r2_targets)]
    A, X = {}, {}
    hh = multi_bfs(adj, anat_handles, n)
    for label, d, r, cset, tg in runs:
        A[label] = analyze(rest, d, r, triangles, adj, cset, anat_handles, tg, canvas_w, shoulder_y)
        X[label] = extras(rest, d, triangles, adj, anat_handles, ring_set, ring_of, rings,
                          A[label]["overlaps"], hh)

    RB = {label: (robust_ratio(rest, d, 0), robust_ratio(rest, d, 1)) for label, d, r, cset, tg in runs}

    def row(label, f):
        print(f"  {label:<46} " + "  ".join(f"{f(k):>16}" for k in (L0, L1, L2)))

    print(f"\n{'='*70}\nHEADLINE COMPARISON (identical code path for all three)\n{'='*70}")
    print(f"  {'':<46} " + "  ".join(f"{k:>16}" for k in (L0, L1, L2)))
    row("max constraint residual (px)", lambda k: f"{A[k]['residual']:.6f}")
    row("inverted triangles (vs own rest orientation)", lambda k: f"{len(A[k]['inverted'])}/{len(triangles)}")
    row("degenerate triangles", lambda k: f"{A[k]['degenerate']}")
    row("precise overlap pairs", lambda k: f"{len(A[k]['overlaps'])}")
    row("max overlap area (px^2)", lambda k: f"{max([o[2] for o in A[k]['overlaps']], default=0):.1f}")
    row("total overlap area (px^2)", lambda k: f"{sum(o[2] for o in A[k]['overlaps']):.1f}")
    row("overlap pairs touching 2 different rings", lambda k: f"{X[k]['cross_ring_overlaps']}")

    print(f"\n{'='*70}\nROTATION FIELD (neighbor angle difference, degrees)\n{'='*70}")
    for k in (L0, L1, L2):
        d = np.array([e[2] for e in A[k]["rot_edges"]])
        print(f"  {k:<7} max={d.max():7.2f}  mean={d.mean():6.2f}  median={np.median(d):6.2f}  "
              f"edges>{BIG_JUMP_DEG:.0f}: {int((d > BIG_JUMP_DEG).sum()):3d}  edges>90: {int((d > 90).sum()):3d}")

    print(f"\n{'='*70}\nEDGE-LENGTH RATIO (deformed/rest)\n{'='*70}")
    pct = [0, 5, 25, 50, 75, 95, 100]
    print(f"  percentile   " + "  ".join(f"{p:>6}" for p in pct))
    for k in (L0, L1, L2):
        print(f"  {k:<10}   " + "  ".join(f"{v:6.3f}" for v in np.percentile(A[k]["ratios"], pct))
              + f"   mean={A[k]['ratios'].mean():.3f}")
    print(f"\n  Mean edge ratio by hop band from each run's OWN constrained set (band 3 = 3+ hops):")
    for k in (L0, L1, L2):
        cells = []
        for b in range(4):
            r = A[k]["band_ratios"][b]
            cells.append(f"b{b}: n={len(r):3d} mean={np.mean(r) if r else float('nan'):.3f} max={max(r) if r else float('nan'):.3f}")
        print(f"  {k:<7} " + " | ".join(cells))

    print(f"\n  Locations of the extreme edges (labels: H=anatomical handle, R=ring member, O=outside ring;")
    print(f"  hh = hops to nearest anatomical handle):")
    for k in (L0, L1, L2):
        lab = X[k]["lab"]
        print(f"  {k} — 5 most COMPRESSED:")
        for r, i, j in X[k]["smallest"]:
            print(f"     ratio={r:6.3f}  edge({i:3d},{j:3d}) {lab(i)}-{lab(j)}  hh=({hh[i]},{hh[j]})  "
                  f"rest=({tuple(np.round(rest[i],0))},{tuple(np.round(rest[j],0))})")
        print(f"  {k} — 5 most STRETCHED:")
        for r, i, j in X[k]["largest"]:
            print(f"     ratio={r:6.3f}  edge({i:3d},{j:3d}) {lab(i)}-{lab(j)}  hh=({hh[i]},{hh[j]})  "
                  f"rest=({tuple(np.round(rest[i],0))},{tuple(np.round(rest[j],0))})")

    print(f"\n{'='*70}\nRING INTERNAL EDGE RATIOS AND RING-TO-RING DISTANCE\n{'='*70}")
    for k in (L0, L1, L2):
        parts = []
        for h in sorted(rings):
            rs = X[k]["internal"][h]
            parts.append(f"{h}:[{min(rs):.2f},{max(rs):.2f}]")
        print(f"  {k:<7} ring-internal edge ratio [min,max] per ring: " + "  ".join(parts))
    print(f"  Min distance between rings, deformed (rest in last column):")
    for (a, b) in sorted(X[L0]["ring_min_dist"]):
        rest_d = min(float(np.linalg.norm(rest[u] - rest[v])) for u in rings[a] for v in rings[b])
        print(f"    ring {a:3d} <-> ring {b:3d}:  " + "  ".join(f"{k}={X[k]['ring_min_dist'][(a,b)]:7.1f}" for k in (L0, L1, L2))
              + f"   rest={rest_d:7.1f}")

    print(f"\n{'='*70}\nBOUNDING BOX AND HEAD-REGION CHECK\n{'='*70}")
    print(f"  {'':<46} " + "  ".join(f"{k:>16}" for k in (L0, L1, L2)))
    row("deformed/rest width ratio", lambda k: f"{A[k]['bbox']['w_ratio']:.3f}")
    row("deformed/rest height ratio", lambda k: f"{A[k]['bbox']['h_ratio']:.3f}")
    print("  -- bbox rows above are informational only; the rule uses the next three rows --")
    row("robust width ratio (P5-P95 spread of x)", lambda k: f"{RB[k][0]:.3f}")
    row("robust height ratio (P5-P95 spread of y)", lambda k: f"{RB[k][1]:.3f}")
    row("far-field mean edge ratio (>=3 hops from handles)", lambda k: f"{X[k]['far_mean']:.3f}")
    row("garment top y (canvas px)", lambda k: f"{A[k]['bbox']['top_y']:.1f}")
    row("garment top above avatar shoulder line (px)", lambda k: f"{A[k]['bbox']['above_shoulder']:.1f}")
    row(f"overflow past canvas right edge x={canvas_w} (px)", lambda k: f"{A[k]['bbox']['right_overflow']:.1f}")

    print(f"\n{'='*70}\nWHERE THE PATHOLOGY LIVES — hops from anatomical handles (0=handle, 1=ring member, 2, 3+)\n{'='*70}")
    for k in (L0, L1, L2):
        big = [e for e in A[k]["rot_edges"] if e[2] > BIG_JUMP_DEG]
        print(f"  {k:<7} overlap pairs hh={hop_hist([o[4] for o in A[k]['overlaps']])}   "
              f"inverted tris hh={hop_hist(A[k]['inv_hops_h'])}   "
              f"rot jumps>{BIG_JUMP_DEG:.0f} hh={hop_hist([e[4] for e in big])}")

    # ---------- registered checks ----------
    print(f"\n{'='*70}\nREGISTERED PREDICTION CHECKS (R2 vs R1, mechanical)\n{'='*70}")
    r1, r2 = A[L1], A[L2]
    d1 = np.array([e[2] for e in r1["rot_edges"]])
    d2 = np.array([e[2] for e in r2["rot_edges"]])
    checks = [
        ("N1 min edge ratio higher than R1", float(r2["ratios"].min()) > float(r1["ratios"].min()),
         f"{r1['ratios'].min():.3f} -> {r2['ratios'].min():.3f}"),
        ("N2 5th-percentile edge ratio higher than R1", float(np.percentile(r2["ratios"], 5)) > float(np.percentile(r1["ratios"], 5)),
         f"{np.percentile(r1['ratios'],5):.3f} -> {np.percentile(r2['ratios'],5):.3f}"),
        ("N3 max rotation jump lower than R1", float(d2.max()) < float(d1.max()),
         f"{d1.max():.2f} -> {d2.max():.2f}"),
        (f"N4 rotation jumps >{BIG_JUMP_DEG:.0f} deg fewer than R1", int((d2 > BIG_JUMP_DEG).sum()) < int((d1 > BIG_JUMP_DEG).sum()),
         f"{int((d1 > BIG_JUMP_DEG).sum())} -> {int((d2 > BIG_JUMP_DEG).sum())}"),
        ("N5 [conditional] overlap pairs fewer than R1", len(r2["overlaps"]) < len(r1["overlaps"]),
         f"{len(r1['overlaps'])} -> {len(r2['overlaps'])}"),
        ("N6 [conditional] inverted triangles fewer than R1", len(r2["inverted"]) < len(r1["inverted"]),
         f"{len(r1['inverted'])} -> {len(r2['inverted'])}"),
    ]
    for label, ok, detail in checks:
        print(f"  {'YES' if ok else 'NO ':3s}  {label:<52} {detail}")

    print(f"\n{'='*70}\nGLOBAL CONTRACTION RULE (MID = {MID:.4f})\n{'='*70}")
    for k in (L0, L1, L2):
        w, h_, f_ = RB[k][0], RB[k][1], X[k]["far_mean"]
        flags = [("width", w), ("height", h_), ("far-field", f_)]
        below = [nm for nm, v in flags if v < MID]
        verdict = ("NO global contraction" if not below else
                   "GLOBAL contraction" if len(below) == 3 else "MIXED (below MID: " + ", ".join(below) + ")")
        print(f"  {k:<7} robust-width={w:.3f}  robust-height={h_:.3f}  far-field={f_:.3f}   -> {verdict}")
    print(f"  (anatomical required scale: shoulder {scale_sh:.3f}, underarm {scale_ua:.3f})")

    # ---------- images: one panel file ----------
    print("\nRendering panel (R2 render | R2 wireframe | R1 wireframe)...")
    garment_arr = np.array(garment.image.convert("RGBA"))
    avatar_rgba = np.array(avatar.image.convert("RGBA"))
    layer = render_mesh(rest, def2, triangles, garment_arr, canvas_w, canvas_h)
    comp = avatar_rgba.copy()
    alpha = layer[:, :, 3:4].astype(float) / 255.0
    comp[:, :, :3] = (avatar_rgba[:, :, :3] * (1 - alpha) + layer[:, :, :3] * alpha).astype(np.uint8)
    left = Image.fromarray(comp).convert("RGB")
    dr = ImageDraw.Draw(left)
    ys, xs = np.where(garment.mask)
    bottom_y = ys.max()
    bxs = xs[ys == bottom_y]
    pts = {"left_hem": ((float(bxs.min()), float(bottom_y)), (255, 0, 0)),
           "right_hem": ((float(bxs.max()), float(bottom_y)), (255, 0, 255)),
           "left_sleeve_end": ((float(xs.min()), float(ys[xs == xs.min()].mean())), (0, 255, 255)),
           "right_sleeve_end": ((float(xs.max()), float(ys[xs == xs.max()].mean())), (0, 255, 0))}
    print("  Diagnostic points (nearest vertex): rest -> ARAP-0 -> R1 -> R2")
    for nm, (pt, color) in pts.items():
        v = int(np.argmin(np.linalg.norm(rest - np.array(pt), axis=1)))
        x, y = def2[v]
        dr.ellipse([x - 6, y - 6, x + 6, y + 6], outline=color, width=3)
        print(f"    {nm:17s} v{v:3d}: {tuple(np.round(rest[v],1))} -> {tuple(np.round(def0[v],1))} -> "
              f"{tuple(np.round(def1[v],1))} -> {tuple(np.round(def2[v],1))}")
    for t in handle_targets.values():
        dr.ellipse([t[0] - 5, t[1] - 5, t[0] + 5, t[1] + 5], outline=(255, 255, 0), width=2)
    dr.text((6, 6), "R2 render", fill=(255, 255, 255))
    mid_img = draw_wire(def2, triangles, ring_set, anat_handles, canvas_w, canvas_h, shoulder_y, "R2 wireframe")
    right_img = draw_wire(def1, triangles, ring_set, anat_handles, canvas_w, canvas_h, shoulder_y, "R1 wireframe")
    panel = Image.new("RGB", (canvas_w * 3, canvas_h), (0, 0, 0))
    panel.paste(left, (0, 0)); panel.paste(mid_img, (canvas_w, 0)); panel.paste(right_img, (canvas_w * 2, 0))
    out_path = f"{OUTPUT_DIR}/checkpoint5_ARAP1_R2_panel.png"
    panel.save(out_path)
    print(f"Saved: {out_path}")
    print("  Wireframes: red = canvas, green = mesh, orange = ring vertices, yellow = handles, cyan = avatar shoulder line.")


if __name__ == "__main__":
    main()