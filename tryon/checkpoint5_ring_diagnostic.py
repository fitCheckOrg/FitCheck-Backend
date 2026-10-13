"""
tryon/checkpoint5_ring_diagnostic.py

READ-ONLY ring diagnostic. No solver, mesh, ring, weighting or target change. R2 and S3
are re-run (deterministic) only so their results can be read.

QUESTION: are the persistent high-stretch ring edges caused by incompatible hard ring
targets generated from the garment's own anatomical-landmark geometry?

Definitions
  constrained edge  both endpoints are ring members -> pinned in R2 AND S3, so its deformed
                    length equals its TARGET length in both runs (verified numerically below).
  same-ring edge    endpoints in the same ring -> by construction ratio = that ring's pair scale
                    (plus the small handle offset).
  cross-ring edge   endpoints in different rings -> ratio is whatever the two pair transforms imply.
  HIGH-STRETCH      S3 deformed/rest edge ratio >= 1.2 (informational threshold, mine).

Registered rule on every HIGH-STRETCH edge found in S3:
  all are cross-ring constrained edges with identical R2/S3 lengths
        -> CONSTRAINT-CONSTRUCTION (hard ring targets)
  any is NOT a constrained edge
        -> NOT EXPLAINED BY HARD TARGETS ALONE (that edge was free to relax: energy/solver implicated)
  none exist
        -> NO HIGH-STRETCH EDGES (nothing to explain on this garment)
Supporting evidence, reported and NOT scored:
  delta = || T_a(x) - T_b(x) ||  at the edge midpoint (the disagreement of the two rings' pair transforms)
  kappa = delta / (mean pair scale * rest length)   (kappa >= 1: target length dominated by the mismatch)
  landmark-level incompatibility: how far each pair transform puts the OTHER pair's landmarks from their targets
  S3 energy by edge category, and left/right ring energy.

Run: python -m tryon.checkpoint5_ring_diagnostic [--garment-ids ID [ID ...]]
Default ids: the golden garment and 831098a3-8778-480f-9a11-aa9df826d0f2.
"""

import argparse
import sys
import numpy as np

from tryon.avatar_representation import build_avatar_representation
from tryon.garment_representation import build_garment_representation
from tryon.piecewise_continuous_test import solve_similarity_2point
from tryon.checkpoint5_ARAP0_deformation_field import (
    build_dense_mesh, build_adjacency, run_arap,
    GOLDEN_GARMENT_ID, AVATAR_USER_ID, MAX_ITERATIONS,
)
from tryon.checkpoint5_ARAP1 import get_anatomical_handles, build_rings, multi_bfs
from tryon.checkpoint5_ARAP1_R2 import pair_by_x, apply_T
from tryon.checkpoint5_S3 import run_arap_scaled, arap_energy, directed_edges

SECOND_GARMENT_ID = "831098a3-8778-480f-9a11-aa9df826d0f2"
HIGH_STRETCH = 1.2
FIXED_TOL = 1e-6

VERDICT_CONSTRUCTION = "CONSTRAINT-CONSTRUCTION (hard ring targets)"
VERDICT_NOT_EXPLAINED = "NOT EXPLAINED BY HARD TARGETS ALONE"
VERDICT_NONE = "NO HIGH-STRETCH EDGES"


def classify_high_stretch(edges):
    """edges: list of dicts with keys 'constrained' (bool), 'cross_ring' (bool), 'fixed_in_both' (bool)."""
    if not edges:
        return VERDICT_NONE
    if all(e["constrained"] and e["cross_ring"] and e["fixed_in_both"] for e in edges):
        return VERDICT_CONSTRUCTION
    return VERDICT_NOT_EXPLAINED


def line_stats(p_small, p_large):
    dx, dy = float(p_large[0] - p_small[0]), float(p_large[1] - p_small[1])
    return dx, dy, float(np.degrees(np.arctan2(dy, dx)))


def diagnose(gid, avatar):
    bar = "=" * 70
    print(f"\n{bar}\nGARMENT {gid}\n{bar}")
    try:
        garment = build_garment_representation(gid, AVATAR_USER_ID)
    except Exception as exc:
        print(f"NOT TESTABLE: garment representation failed: {type(exc).__name__}: {exc}")
        return None
    if not garment.shoulders_available:
        print("NOT TESTABLE: shoulders_available=False (four-handle design needs shoulders).")
        return None
    try:
        rest, triangles, landmark_index, n_boundary = build_dense_mesh(garment)
    except Exception as exc:
        print(f"NOT TESTABLE: mesh construction failed: {type(exc).__name__}: {exc}")
        return None
    n = len(rest)
    adj = build_adjacency(n, triangles)
    handle_targets = get_anatomical_handles(garment, avatar, landmark_index)
    anat = set(handle_targets)
    hh = multi_bfs(adj, anat, n)

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

    name_of, T_of, scale_of = {}, {}, {}
    for nm in ("left_underarm", "right_underarm", "left_shoulder", "right_shoulder"):
        v = landmark_index[nm]
        name_of[v] = nm
        T_of[v] = T_ua if "underarm" in nm else T_sh
        scale_of[v] = scale_ua if "underarm" in nm else scale_sh

    rings, problems = build_rings(adj, handle_targets)
    if problems:
        print("NOT TESTABLE: ring validation failed: " + "; ".join(problems))
        return None
    ring_set = set().union(*rings.values())
    ring_of = {v: h for h, ring in rings.items() for v in ring}
    r2_targets = {}
    for h, ring in rings.items():
        for v in ring:
            r2_targets[v] = handle_targets[h].copy() if v == h else apply_T(T_of[h], rest[v])

    xs_h = np.array([rest[h][0] for h in anat])
    side_of = {h: ("small-x" if rest[h][0] < xs_h.mean() else "large-x") for h in anat}

    print(f"Mesh: {n} vertices, {len(triangles)} triangles; handles {sorted(anat)}; "
          f"ring sizes " + ", ".join(f"{h}:{len(r)}" for h, r in sorted(rings.items())))

    # -------- solves (read-only use of the results) --------
    def_r2, rot_r2 = run_arap(rest, adj, ring_set, r2_targets, MAX_ITERATIONS)
    def_s3, rot_s3, s_final, trace, clamp_hits = run_arap_scaled(rest, adj, ring_set, r2_targets, MAX_ITERATIONS)
    I, J = directed_edges(adj)
    E_s3 = arap_energy(rest, def_s3, rot_s3, s_final, I, J)
    E_r2 = arap_energy(rest, def_r2, rot_r2, 1.0, I, J)
    rel = abs(E_s3 - trace[-1][1]) / trace[-1][1]
    print(f"S3 reproduced: final s = {s_final:.5f}, final E = {E_s3:.4f}   "
          f"(matches the S3 run's trace: relative gap {rel:.1e} -> {'PASS' if rel < 1e-9 else 'FAIL'})")
    print(f"R2 reproduced: E(s=1) = {E_r2:.4f}")

    # ================= 1. DETECTED GEOMETRY =================
    print(f"\n{'-'*70}\n1. DETECTED GARMENT / AVATAR GEOMETRY\n{'-'*70}")
    for label, sh, ua in (("garment", g_sh, g_ua), ("avatar ", a_sh, a_ua)):
        dxs, dys, angs = line_stats(sh[0], sh[1])
        dxu, dyu, angu = line_stats(ua[0], ua[1])
        print(f"  {label} shoulders small-x=({sh[0][0]:.1f},{sh[0][1]:.1f}) large-x=({sh[1][0]:.1f},{sh[1][1]:.1f})  "
              f"span={dxs:.1f}  VERTICAL SEPARATION={dys:+.1f}px  line angle={angs:+.2f} deg")
        print(f"  {label} underarms small-x=({ua[0][0]:.1f},{ua[0][1]:.1f}) large-x=({ua[1][0]:.1f},{ua[1][1]:.1f})  "
              f"span={dxu:.1f}  vertical separation={dyu:+.1f}px  line angle={angu:+.2f} deg")
        print(f"  {label} shoulder->underarm vertical distance: small-x side={ua[0][1]-sh[0][1]:.1f}px  large-x side={ua[1][1]-sh[1][1]:.1f}px")
    print(f"  pair transforms: T_underarm scale={scale_ua:.4f} rot={rot_ua:+.2f} deg | "
          f"T_shoulder scale={scale_sh:.4f} rot={rot_sh:+.2f} deg | rotation difference={rot_ua - rot_sh:+.2f} deg | "
          f"scale ratio shoulder/underarm={scale_sh / scale_ua:.3f}")
    e_sh = [float(np.linalg.norm(apply_T(T_ua, g_sh[k]) - np.array(a_sh[k], float))) for k in (0, 1)]
    e_ua = [float(np.linalg.norm(apply_T(T_sh, g_ua[k]) - np.array(a_ua[k], float))) for k in (0, 1)]
    print(f"  LANDMARK-LEVEL INCOMPATIBILITY: T_underarm puts the garment's SHOULDERS {e_sh[0]:.1f}px / {e_sh[1]:.1f}px "
          f"(small-x / large-x) from the avatar's shoulders;")
    print(f"                                  T_shoulder puts the garment's UNDERARMS {e_ua[0]:.1f}px / {e_ua[1]:.1f}px from the avatar's underarms.")

    # ================= 2. RING MAP =================
    print(f"\n{'-'*70}\n2. RING MAP AND RING-TO-RING DISTANCES\n{'-'*70}")
    for h in sorted(rings, key=lambda k: rest[k][0]):
        mem = "  ".join(f"{v}:({rest[v][0]:.0f},{rest[v][1]:.0f})->({r2_targets[v][0]:.0f},{r2_targets[v][1]:.0f})" for v in sorted(rings[h]))
        print(f"  ring {h:3d} {name_of[h]:<15} side={side_of[h]}  pair scale={scale_of[h]:.3f}  members rest->target: {mem}")
    hs = sorted(rings)
    pair_rows = []
    for ia in range(len(hs)):
        for ib in range(ia + 1, len(hs)):
            a, b = hs[ia], hs[ib]
            dist = multi_bfs(adj, rings[a], n)
            hops = min(dist[v] for v in rings[b])
            u0, v0 = min(((u, v) for u in rings[a] for v in rings[b]), key=lambda p: np.linalg.norm(rest[p[0]] - rest[p[1]]))
            d_rest = float(np.linalg.norm(rest[u0] - rest[v0]))
            d_tgt = float(np.linalg.norm(r2_targets[u0] - r2_targets[v0]))
            mid = (rest[u0] + rest[v0]) / 2
            delta = float(np.linalg.norm(apply_T(T_of[a], mid) - apply_T(T_of[b], mid)))
            pair_rows.append((a, b, hops, d_rest, d_tgt, delta))
            print(f"  ring {a:3d} <-> ring {b:3d}: {hops} hop(s) apart; closest at rest = vertices ({u0},{v0}) rest {d_rest:6.1f}px -> "
                  f"target {d_tgt:6.1f}px (ratio {d_tgt / d_rest:.3f}); transform disagreement there delta={delta:.1f}px"
                  f"   [{'SAME transform' if T_of[a] is T_of[b] else 'DIFFERENT transforms'}]")

    # ================= 3. CONSTRAINED EDGES =================
    print(f"\n{'-'*70}\n3. CONSTRAINED EDGES (both endpoints pinned) AND TARGET CONSISTENCY\n{'-'*70}")
    cons = []
    for i in adj:
        for j in adj[i]:
            if j > i and i in ring_of and j in ring_of:
                rl = float(np.linalg.norm(rest[i] - rest[j]))
                tl = float(np.linalg.norm(r2_targets[i] - r2_targets[j]))
                l2 = float(np.linalg.norm(def_r2[i] - def_r2[j]))
                l3 = float(np.linalg.norm(def_s3[i] - def_s3[j]))
                mid = (rest[i] + rest[j]) / 2
                delta = float(np.linalg.norm(apply_T(T_of[ring_of[i]], mid) - apply_T(T_of[ring_of[j]], mid)))
                ms = (scale_of[ring_of[i]] + scale_of[ring_of[j]]) / 2
                cons.append(dict(i=i, j=j, ri=ring_of[i], rj=ring_of[j], rest=rl, tgt=tl, ratio=tl / rl,
                                 same=ring_of[i] == ring_of[j], delta=delta, kappa=delta / (ms * rl),
                                 fixed_err=max(abs(l2 - tl), abs(l3 - tl))))
    max_fixed_err = max((c["fixed_err"] for c in cons), default=0.0)
    print(f"  {len(cons)} constrained edges. Deformed length == target length in BOTH R2 and S3 for all of them: "
          f"max |difference| = {max_fixed_err:.2e} px ({'VERIFIED' if max_fixed_err < FIXED_TOL else 'NOT VERIFIED'})")
    same = [c for c in cons if c["same"]]
    cross = sorted([c for c in cons if not c["same"]], key=lambda c: -c["ratio"])
    for h in sorted(rings):
        rr = [c["ratio"] for c in same if c["ri"] == h]
        if rr:
            print(f"  same-ring edges, ring {h:3d} ({name_of[h]:<15}): n={len(rr)} target/rest ratio min={min(rr):.3f} "
                  f"median={np.median(rr):.3f} max={max(rr):.3f}   (pair scale {scale_of[h]:.3f})")
    print(f"  CROSS-RING constrained edges ({len(cross)}), sorted by target/rest ratio:")
    if not cross:
        print("    none")
    for c in cross:
        print(f"    edge({c['i']:3d},{c['j']:3d}) rings=({c['ri']},{c['rj']}) rest={c['rest']:6.1f}px target={c['tgt']:6.1f}px "
              f"ratio={c['ratio']:.3f}  transform disagreement delta={c['delta']:6.1f}px  kappa={c['kappa']:.2f}")

    # ================= 4. EXTREME EDGES WITH FULL ATTRIBUTION =================
    print(f"\n{'-'*70}\n4. EXTREME EDGES (R2 and S3), FULL ATTRIBUTION\n{'-'*70}")
    def lab(v):
        return "H" if v in anat else ("R" if v in ring_set else "O")
    def cat(i, j):
        ri, rj = ring_of.get(i), ring_of.get(j)
        if ri is not None and rj is not None:
            return "same-ring" if ri == rj else "CROSS-RING"
        return "ring-boundary" if (ri is not None or rj is not None) else "free"
    def all_edges(cur):
        out = []
        for i in adj:
            for j in adj[i]:
                if j > i:
                    rl = float(np.linalg.norm(rest[i] - rest[j]))
                    if rl > 1e-9:
                        out.append((float(np.linalg.norm(cur[i] - cur[j])) / rl, i, j, rl))
        return sorted(out)
    E2, E3 = all_edges(def_r2), all_edges(def_s3)
    ratio2 = {(i, j): r for r, i, j, _ in E2}
    def show(tag, rows, cur_other):
        for r, i, j, rl in rows:
            r_other = ratio2[(i, j)] if cur_other == "R2" else None
            fixed = (r_other is not None and abs(r - r_other) < FIXED_TOL)
            extra = f"  R2 ratio={r_other:.3f}{' (IDENTICAL -> pinned)' if fixed else ''}" if r_other is not None else ""
            print(f"    {tag} ratio={r:6.3f} edge({i:3d},{j:3d}) {lab(i)}-{lab(j)} rings=({ring_of.get(i, '-')},{ring_of.get(j, '-')}) "
                  f"hh=({hh[i]},{hh[j]}) {cat(i, j):<13} rest=({rest[i][0]:.0f},{rest[i][1]:.0f})-({rest[j][0]:.0f},{rest[j][1]:.0f}) len={rl:.1f}px{extra}")
    print("  R2 — 5 most COMPRESSED:"); show("R2", E2[:5], None)
    print("  R2 — 5 most STRETCHED:");  show("R2", E2[-5:][::-1], None)
    print("  S3 — 5 most COMPRESSED:"); show("S3", E3[:5], "R2")
    print("  S3 — 5 most STRETCHED:");  show("S3", E3[-5:][::-1], "R2")

    # ================= 5. HIGH-STRETCH VERDICT =================
    print(f"\n{'-'*70}\n5. HIGH-STRETCH EDGES IN S3 (ratio >= {HIGH_STRETCH}) AND THE REGISTERED RULE\n{'-'*70}")
    cons_by = {(c["i"], c["j"]): c for c in cons}
    hs_edges = []
    for r, i, j, rl in E3:
        if r >= HIGH_STRETCH:
            c = cons_by.get((i, j))
            hs_edges.append(dict(i=i, j=j, ratio=r, constrained=c is not None,
                                 cross_ring=bool(c and not c["same"]),
                                 fixed_in_both=abs(r - ratio2[(i, j)]) < FIXED_TOL))
            tag = "constrained, " + ("CROSS-RING" if c and not c["same"] else "same-ring") if c else f"NOT constrained ({cat(i, j)})"
            extra = f"  delta={c['delta']:.1f}px kappa={c['kappa']:.2f} rings=({c['ri']},{c['rj']})" if c else ""
            print(f"    edge({i},{j}) S3 ratio={r:.3f}  R2 ratio={ratio2[(i, j)]:.3f}  {tag}{extra}")
    verdict = classify_high_stretch(hs_edges)
    print(f"  edges with S3 ratio >= {HIGH_STRETCH}: {len(hs_edges)}")
    print(f"  VERDICT: {verdict}")

    # ================= 6. ENERGY =================
    print(f"\n{'-'*70}\n6. ENERGY BY EDGE CATEGORY, PER RING, AND LEFT/RIGHT\n{'-'*70}")
    def breakdown(cur, R, s):
        cats = {"same-ring": 0.0, "CROSS-RING": 0.0, "ring-boundary": 0.0, "free": 0.0}
        ring_comp = {h: {"internal": 0.0, "cross": 0.0, "boundary": 0.0} for h in rings}
        for i in adj:
            for j in adj[i]:
                d = (cur[i] - cur[j]) - s * (R[i] @ (rest[i] - rest[j]))
                e = float(d @ d)
                cats[cat(i, j)] += e
                if i in ring_of:
                    k = "internal" if ring_of.get(j) == ring_of[i] else ("cross" if j in ring_of else "boundary")
                    ring_comp[ring_of[i]][k] += e
        return cats, ring_comp
    cats2, rc2 = breakdown(def_r2, rot_r2, 1.0)
    cats3, rc3 = breakdown(def_s3, rot_s3, s_final)
    print(f"  integrity: category energies sum to total? R2 {sum(cats2.values()):.4f} vs {E_r2:.4f}; S3 {sum(cats3.values()):.4f} vs {E_s3:.4f}"
          f"  -> {'PASS' if abs(sum(cats3.values()) - E_s3) < 1e-6 * E_s3 and abs(sum(cats2.values()) - E_r2) < 1e-6 * E_r2 else 'FAIL'}")
    print(f"  {'category':<16}{'R2 energy':>14}{'R2 share':>10}{'S3 energy':>14}{'S3 share':>10}")
    for k in cats3:
        print(f"  {k:<16}{cats2[k]:14.1f}{cats2[k] / E_r2:10.1%}{cats3[k]:14.1f}{cats3[k] / E_s3:10.1%}")
    print(f"\n  per-ring energy (directed edges owned by the ring's vertices), components internal / cross-ring / boundary:")
    print(f"  {'ring':<22}{'side':<9}{'R2 total':>10}{'S3 total':>10}   S3 internal / cross / boundary")
    side_tot = {"small-x": [0.0, 0.0], "large-x": [0.0, 0.0]}
    for h in sorted(rings, key=lambda k: rest[k][0]):
        t2, t3 = sum(rc2[h].values()), sum(rc3[h].values())
        side_tot[side_of[h]][0] += t2
        side_tot[side_of[h]][1] += t3
        print(f"  {h:3d} {name_of[h]:<17}{side_of[h]:<9}{t2:10.1f}{t3:10.1f}   "
              f"{rc3[h]['internal']:8.1f} / {rc3[h]['cross']:8.1f} / {rc3[h]['boundary']:8.1f}")
    lr = {k: (side_tot[k][1] / side_tot["large-x"][1] if side_tot["large-x"][1] > 0 else float("nan")) for k in side_tot}
    print(f"  LEFT/RIGHT totals (ring-owned energy): small-x R2={side_tot['small-x'][0]:.1f} S3={side_tot['small-x'][1]:.1f} | "
          f"large-x R2={side_tot['large-x'][0]:.1f} S3={side_tot['large-x'][1]:.1f} | S3 small-x/large-x = {lr['small-x']:.2f}x")

    return dict(gid=gid, s=s_final, E=E_s3, e_sh=e_sh, e_ua=e_ua, rot_diff=rot_ua - rot_sh,
                shoulder_dy=line_stats(g_sh[0], g_sh[1])[1], underarm_dy=line_stats(g_ua[0], g_ua[1])[1],
                scale_ratio=scale_sh / scale_ua, n_cross=len(cross),
                max_cross=max((c["ratio"] for c in cross), default=float("nan")),
                max_kappa=max((c["kappa"] for c in cross), default=float("nan")),
                cross_share=cats3["CROSS-RING"] / E_s3, lr=lr["small-x"], n_hs=len(hs_edges), verdict=verdict)


def main(ids):
    print(__doc__)
    print("Building the avatar once...")
    try:
        avatar = build_avatar_representation()
    except Exception as exc:
        print(f"avatar build failed: {type(exc).__name__}: {exc}")
        sys.exit(2)
    results = [r for r in (diagnose(g, avatar) for g in ids) if r is not None]
    if not results:
        print("\nNo garment was testable.")
        sys.exit(2)
    bar = "=" * 70
    print(f"\n{bar}\nCROSS-GARMENT SUMMARY (observations; no mechanism change implied)\n{bar}")
    rows = [
        ("garment", lambda r: r["gid"][:8]),
        ("final s", lambda r: f"{r['s']:.4f}"),
        ("garment shoulder vertical separation (px)", lambda r: f"{r['shoulder_dy']:+.1f}"),
        ("garment underarm vertical separation (px)", lambda r: f"{r['underarm_dy']:+.1f}"),
        ("pair rotation difference (deg)", lambda r: f"{r['rot_diff']:+.2f}"),
        ("scale ratio shoulder/underarm", lambda r: f"{r['scale_ratio']:.3f}"),
        ("T_ua error at garment shoulders (px) small/large-x", lambda r: f"{r['e_sh'][0]:.1f} / {r['e_sh'][1]:.1f}"),
        ("T_sh error at garment underarms (px) small/large-x", lambda r: f"{r['e_ua'][0]:.1f} / {r['e_ua'][1]:.1f}"),
        ("cross-ring constrained edges", lambda r: f"{r['n_cross']}"),
        ("max cross-ring target/rest ratio", lambda r: f"{r['max_cross']:.3f}"),
        ("max kappa on cross-ring edges", lambda r: f"{r['max_kappa']:.2f}"),
        ("S3 energy share in cross-ring edges", lambda r: f"{r['cross_share']:.1%}"),
        ("S3 ring energy small-x / large-x", lambda r: f"{r['lr']:.2f}x"),
        ("S3 edges with ratio >= 1.2", lambda r: f"{r['n_hs']}"),
        ("registered verdict", lambda r: r["verdict"]),
    ]
    for label, f in rows:
        print(f"  {label:<54}" + "".join(f"{f(r):>46}" if label == "registered verdict" else f"{f(r):>16}" for r in results))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Read-only ring diagnostic over one or more garments.")
    ap.add_argument("--garment-ids", nargs="*", default=[GOLDEN_GARMENT_ID, SECOND_GARMENT_ID])
    args = ap.parse_args()
    main(args.garment_ids)