"""
tryon/checkpoint5_S3.py

S3 — ARAP with ONE global scale DOF, on the R2 rings.

ENERGY (the only thing that changes vs the frozen R2 run):
    E(p', R, s) = sum_i sum_{j in N(i)} || (p'_i - p'_j) - s * R_i (p_i - p_j) ||^2
with a single scalar s > 0 shared by the whole mesh.

SOLVER: block-coordinate descent, each block an EXACT minimizer, so E cannot increase:
    1. local step   : R_i from SVD (identical to the frozen solver; s>0 does not change it)
    2. scale step   : s* = sum <p'_i - p'_j, R_i (p_i - p_j)> / sum ||p_i - p_j||^2   (closed form)
                      clamped to [S_MIN, S_MAX]; clamp activations are counted and reported.
                      s is NEVER set by hand to 0.303, 0.407 or their average.
    3. global step  : same sparse solve as the frozen solver, free-vertex right-hand side
                      multiplied by s; constrained rows still hold their hard targets.
Initial s = 1 (so iteration 1 starts exactly like the frozen R2 run).

FROZEN BY IMPORT: mesh, adjacency, uniform weights, cold start, local step, Laplacian,
iteration cap, overlap detector, ring construction, R2 target construction rule, and
the R2-script analysis helpers. R2 (s fixed at 1) is re-run in-process as the baseline.

INTEGRITY CHECKS built in:
    - the S3 solver with s frozen at 1 must reproduce the frozen R2 run (max |diff| ~ 0)
    - E must be non-increasing across iterations
    - final s-step residual (|s*(final) - s| / s) as a convergence indicator

Run: python -m tryon.checkpoint5_S3
"""

import argparse
import os
import sys
import numpy as np
import scipy.sparse.linalg as spla
from PIL import Image, ImageDraw

from tryon.avatar_representation import build_avatar_representation
from tryon.garment_representation import build_garment_representation
from tryon.piecewise_continuous_test import solve_similarity_2point
from tryon.checkpoint5_ARAP0_deformation_field import (
    build_dense_mesh, build_adjacency, build_laplacian, local_step, run_arap,
    GOLDEN_GARMENT_ID, AVATAR_USER_ID, MAX_ITERATIONS,
)
from tryon.checkpoint5_ARAP1 import (
    get_anatomical_handles, build_rings, analyze, hop_hist, multi_bfs, render_mesh, BIG_JUMP_DEG,
)
from tryon.checkpoint5_ARAP1_R2 import pair_by_x, robust_ratio, apply_T, extras, draw_wire

OUTPUT_DIR = "outputs"
S_MIN, S_MAX = 0.05, 2.0  # guard rails far outside the expected range; clamp hits are reported

REGISTERED = """
======================================================================
REGISTERED BEFORE RUNNING (S3)
======================================================================
Experiment claim: S3 asks whether ONE global scale DOF in ARAP's energy lets ARAP carry the
required contraction into the unconstrained garment, even though the anatomical constraints
demand two different local scales (shoulder ~0.303, underarm ~0.407). Same R2 rings, mesh,
uniform weights, cold start and iteration cap. s comes from the ARAP objective, not from the handles.

Key question: does adding s change the UNCONSTRAINED (>= 3 hops from every handle) region?
(Overlap counts are NOT the key diagnostic; R2 already fixed most of that.)

Outcome rule (same quantities and MID line as R2; MID = (1 + max pair scale)/2):
    robust width ratio, robust height ratio, far-field mean edge ratio
    all three < MID                      -> CONTRACTION   (the missing scale DOF was a real limitation)
    all three > MID and far-field mean
      within 0.05 of R2's                -> NO CHANGE     (one global s does not move the far field)
    anything else                        -> PARTIAL / MIXED (printed with which quantities moved)
Informational (registered, not pass/fail):
    where final s sits relative to the required band [0.303, 0.407]: below / inside / above
    per-ring conflict: a ring whose pair scale differs from s by > 15% conflicts with the single s
      (that pattern would indicate FitCheck needs spatially varying scale, not that S3 failed)
    collar / hem placement relative to the T_underarm baseline (printed, not scored)
A far field that contracts does NOT imply the garment looks right: sleeve/hem semantics
(Checkpoint 3) are a separate problem.
"""


GENERALIZATION_RULE = """
======================================================================
GENERALIZATION RULE (registered before any second-garment data existed)
======================================================================
Same S3 implementation, same avatar, same R2 ring construction, same eight checks;
only the garment differs. NOTE: the scale values quoted in the claim above belong to the
golden pair; this garment's own pair scales and MID are printed below.
  PASS          outcome is CONTRACTION (all three quantities below THIS garment's MID)
                AND handle residual < 1e-6
                AND S3 overlap pairs <= R2's AND S3 inverted triangles <= R2's
                AND S3 max neighbor rotation jump <= R2's
                AND energy never increases (solver integrity)
  PARTIAL       far-field mean edge ratio is below MID but one or more PASS conditions fail
                (scale works; the failing items are named = another limitation exposed)
  FAIL          far-field mean edge ratio is NOT below MID (the global-scale mechanism did not carry over)
  NOT TESTABLE  the garment/ring construction cannot be built (no shoulders, detection error,
                ring validation failure). Reported; NOT counted against S3.
  Informational only (not scored): where s lands vs this garment's pair-scale band,
  per-ring conflicts, collar/hem placement, top-above-shoulder distance.
"""


def classify_generalization(contraction, far_ok, res_ok, ov_ok, inv_ok, rot_ok, integrity_ok):
    failing = []
    if not res_ok:
        failing.append("handle residual not ~0")
    if not integrity_ok:
        failing.append("energy increased (solver integrity)")
    if not ov_ok:
        failing.append("S3 overlap pairs exceed R2's")
    if not inv_ok:
        failing.append("S3 inverted triangles exceed R2's")
    if not rot_ok:
        failing.append("S3 max rotation jump exceeds R2's")
    if not res_ok or not integrity_ok:
        return "INVALID RUN", failing
    if not far_ok:
        return "FAIL", ["far-field mean edge ratio is not below MID"] + failing
    if contraction and not failing:
        return "PASS", []
    if not failing:
        failing = ["far field contracted but robust width and height are not both below MID"]
    return "PARTIAL", failing


def directed_edges(adj):
    I, J = [], []
    for i in sorted(adj):
        for j in adj[i]:
            I.append(i)
            J.append(j)
    return np.array(I), np.array(J)


def arap_energy(rest, cur, R, s, I, J):
    e_rest = rest[I] - rest[J]
    e_cur = cur[I] - cur[J]
    rot = np.einsum('mab,mb->ma', R[I], e_rest)
    diff = e_cur - s * rot
    return float((diff ** 2).sum())


def s_star(rest, cur, R, I, J):
    e_rest = rest[I] - rest[J]
    rot = np.einsum('mab,mb->ma', R[I], e_rest)
    return float(((cur[I] - cur[J]) * rot).sum() / (e_rest ** 2).sum())


def global_step_scaled(Lf, rest, adj, R, n, constrained, targets, s):
    bx, by = np.zeros(n), np.zeros(n)
    for i in range(n):
        if i in constrained:
            bx[i], by[i] = targets[i]
            continue
        acc = np.zeros(2)
        for j in adj[i]:
            acc += 0.5 * (R[i] + R[j]) @ (rest[i] - rest[j])
        bx[i], by[i] = s * acc
    return np.stack([Lf(bx), Lf(by)], axis=1)


def run_arap_scaled(rest, adj, constrained, targets, max_iterations, freeze_s=False, s_init=1.0):
    n = len(rest)
    cur = rest.copy()
    for v in constrained:
        cur[v] = targets[v]
    I, J = directed_edges(adj)
    L = build_laplacian(n, adj, constrained)
    Lf = spla.factorized(L.tocsc())
    s = float(s_init)
    clamp_hits = 0
    trace = []
    R = None
    for it in range(1, max_iterations + 1):
        R = local_step(rest, cur, adj, n)
        if not freeze_s:
            raw = s_star(rest, cur, R, I, J)
            s_new = min(max(raw, S_MIN), S_MAX)
            if s_new != raw:
                clamp_hits += 1
            s = s_new
        new_cur = global_step_scaled(Lf, rest, adj, R, n, constrained, targets, s)
        max_disp = float(np.max(np.linalg.norm(new_cur - cur, axis=1)))
        cur = new_cur
        trace.append((it, arap_energy(rest, cur, R, s, I, J), s, max_disp))
    return cur, R, s, trace, clamp_hits


def far_ratios(rest, deformed, adj, hh):
    out = []
    for i in adj:
        for j in adj[i]:
            if j > i and min(hh[i], hh[j]) >= 3:
                rl = float(np.linalg.norm(rest[i] - rest[j]))
                if rl > 1e-9:
                    out.append(float(np.linalg.norm(deformed[i] - deformed[j])) / rl)
    return np.array(out)


def ring_energy(rest, cur, R, s, adj, ring):
    tot = 0.0
    for i in ring:
        for j in adj[i]:
            d = (cur[i] - cur[j]) - s * (R[i] @ (rest[i] - rest[j]))
            tot += float(d @ d)
    return tot


def main(garment_id=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(REGISTERED)
    gid = garment_id or GOLDEN_GARMENT_ID
    override = gid != GOLDEN_GARMENT_ID
    if override:
        print(GENERALIZATION_RULE)
    print(f"GARMENT UNDER TEST: {gid}" + ("   (OVERRIDE of the golden garment)" if override else "   (golden garment, default)"))

    print("Rebuilding real AvatarRepresentation and GarmentRepresentation...")
    try:
        avatar = build_avatar_representation()
        garment = build_garment_representation(gid, AVATAR_USER_ID)
    except Exception as exc:
        print(f"\nNOT TESTABLE: garment/avatar representation failed: {type(exc).__name__}: {exc}")
        print("  (a setup incompatibility, not counted against S3)")
        sys.exit(2)
    if not garment.shoulders_available:
        print("\nNOT TESTABLE: this garment has shoulders_available=False. The four-handle design needs shoulder")
        print("  landmarks; the two-handle case (ARAP-0b) is a separate, deferred experiment. Not counted against S3.")
        sys.exit(2)
    print(f"  garment image size={garment.image.size}  underarms L={tuple(garment.left_underarm)} R={tuple(garment.right_underarm)}"
          f"  shoulders L={tuple(garment.left_shoulder)} R={tuple(garment.right_shoulder)}")

    try:
        rest, triangles, landmark_index, n_boundary = build_dense_mesh(garment)
    except Exception as exc:
        print(f"\nNOT TESTABLE: mesh construction failed: {type(exc).__name__}: {exc}")
        print("  (a setup incompatibility, not counted against S3)")
        sys.exit(2)
    n = len(rest)
    adj = build_adjacency(n, triangles)
    handle_targets = get_anatomical_handles(garment, avatar, landmark_index)
    anat_handles = set(handle_targets.keys())
    canvas_w, canvas_h = avatar.image.size
    shoulder_y = float(min(avatar.left_shoulder[1], avatar.right_shoulder[1]))
    print(f"Mesh: {n} vertices, {len(triangles)} triangles. Anatomical handles: {sorted(anat_handles)}")

    # ---- pair transforms and R2 targets: identical construction to the R2 script ----
    g_ua = pair_by_x(garment.left_underarm, garment.right_underarm)
    g_sh = pair_by_x(garment.left_shoulder, garment.right_shoulder)
    a_ua = pair_by_x(avatar.left_underarm, avatar.right_underarm)
    a_sh = pair_by_x(avatar.left_shoulder, avatar.right_shoulder)
    T_ua = np.array(solve_similarity_2point(list(g_ua), list(a_ua)))
    T_sh = np.array(solve_similarity_2point(list(g_sh), list(a_sh)))
    scale_ua = float(np.hypot(T_ua[0, 0], T_ua[1, 0]))
    scale_sh = float(np.hypot(T_sh[0, 0], T_sh[1, 0]))
    MID = (1.0 + max(scale_ua, scale_sh)) / 2.0
    lo, hi = sorted((scale_sh, scale_ua))
    print(f"T_underarm scale={scale_ua:.4f}  T_shoulder scale={scale_sh:.4f}  MID={MID:.4f}  required band=[{lo:.3f}, {hi:.3f}]")

    names = ("left_underarm", "right_underarm", "left_shoulder", "right_shoulder")
    name_of, T_of = {}, {}
    for nm in names:
        v = landmark_index[nm]
        name_of[v] = nm
        T_of[v] = T_ua if "underarm" in nm else T_sh

    rings, problems = build_rings(adj, handle_targets)
    if problems:
        print("RING VALIDATION FAILED — not resolving silently:")
        for p in problems:
            print(f"  - {p}")
        print("NOT TESTABLE: the R2 ring construction is ambiguous on this mesh (setup incompatibility, not counted against S3).")
        sys.exit(2)
    ring_set = set().union(*rings.values())
    ring_of = {v: h for h, ring in rings.items() for v in ring}
    r2_targets = {}
    for h, ring in rings.items():
        for v in ring:
            r2_targets[v] = handle_targets[h].copy() if v == h else apply_T(T_of[h], rest[v])
    print(f"R2 rings rebuilt (sizes " + ", ".join(f"{h}:{len(r)}" for h, r in sorted(rings.items()))
          + f", {len(ring_set)} constrained vertices).")

    # ---- solves ----
    print(f"\nRunning frozen R2 baseline ({MAX_ITERATIONS} it, s fixed at 1)...")
    def_r2, rot_r2 = run_arap(rest, adj, ring_set, r2_targets, MAX_ITERATIONS)

    print("Integrity check 1: S3 solver with s frozen at 1 vs the frozen R2 run...")
    chk, _, _, _, _ = run_arap_scaled(rest, adj, ring_set, r2_targets, MAX_ITERATIONS, freeze_s=True)
    max_diff = float(np.abs(chk - def_r2).max())
    print(f"  max |position difference| = {max_diff:.3e} px   ({'PASS' if max_diff < 1e-6 else 'FAIL — S3 solver is NOT equivalent to the frozen solver at s=1'})")
    if max_diff >= 1e-6:
        print("  Stopping: S3 results would not be attributable to s alone.")
        sys.exit(1)

    print(f"Running S3 ({MAX_ITERATIONS} it, s free)...")
    def_s3, rot_s3, s_final, trace, clamp_hits = run_arap_scaled(rest, adj, ring_set, r2_targets, MAX_ITERATIONS)

    # ---- convergence trace ----
    print(f"\n{'='*70}\nS3 CONVERGENCE TRACE (energy after each full iteration)\n{'='*70}")
    prev = None
    first_below = {1e-3: None, 1e-4: None, 1e-5: None}
    increases = 0
    for it, E, s, md in trace:
        rel = abs(E - prev) / max(prev, 1e-12) if prev is not None else float("nan")
        if prev is not None:
            if E > prev * (1 + 1e-9):
                increases += 1
            for t in first_below:
                if first_below[t] is None and rel < t:
                    first_below[t] = it
        print(f"  iter {it:3d}  E={E:14.4f}  dE_rel={rel if it > 1 else float('nan'):.6f}  s={s:.5f}  max_disp={md:.4f}px")
        prev = E
    print(f"\n  Integrity check 2: iterations where E INCREASED = {increases}   ({'PASS' if increases == 0 else 'FAIL — block updates are not exact minimizers'})")
    print(f"  s clamp activations (bounds [{S_MIN}, {S_MAX}]) = {clamp_hits}")
    print(f"  first iteration with dE_rel < 1e-3 / 1e-4 / 1e-5: {first_below[1e-3]} / {first_below[1e-4]} / {first_below[1e-5]}")
    I, J = directed_edges(adj)
    R_fin = local_step(rest, def_s3, adj, n)
    s_chk = s_star(rest, def_s3, R_fin, I, J)
    print(f"  Integrity check 3: final s = {s_final:.5f}; s-step re-solved on the final configuration = {s_chk:.5f}  "
          f"(relative gap {abs(s_chk - s_final) / s_final:.2e})")
    s_band = "BELOW the required band" if s_final < lo else ("INSIDE the required band" if s_final <= hi else "ABOVE the required band")
    print(f"  final s = {s_final:.4f} -> {s_band} [{lo:.3f}, {hi:.3f}]")

    # ---- analysis (identical code path) ----
    L2, L3 = "R2", "S3"
    runs = [(L2, def_r2, rot_r2, 1.0), (L3, def_s3, rot_s3, s_final)]
    hh = multi_bfs(adj, anat_handles, n)
    A, X, RB = {}, {}, {}
    for label, d, r, s in runs:
        A[label] = analyze(rest, d, r, triangles, adj, ring_set, anat_handles, r2_targets, canvas_w, shoulder_y)
        X[label] = extras(rest, d, triangles, adj, anat_handles, ring_set, ring_of, rings, A[label]["overlaps"], hh)
        RB[label] = (robust_ratio(rest, d, 0), robust_ratio(rest, d, 1))

    def row(label, f):
        print(f"  {label:<50} " + "  ".join(f"{f(k):>14}" for k in (L2, L3)))

    print(f"\n{'='*70}\nHEADLINE COMPARISON (R2 baseline vs S3; identical code path)\n{'='*70}")
    print(f"  {'':<50} " + "  ".join(f"{k:>14}" for k in (L2, L3)))
    row("global scale s used", lambda k: f"{dict((a, b) for a, _, _, b in runs)[k]:.4f}")
    row("max constraint residual (px)", lambda k: f"{A[k]['residual']:.6f}")
    row("inverted triangles", lambda k: f"{len(A[k]['inverted'])}/{len(triangles)}")
    row("degenerate triangles", lambda k: f"{A[k]['degenerate']}")
    row("precise overlap pairs", lambda k: f"{len(A[k]['overlaps'])}")
    row("max overlap area (px^2)", lambda k: f"{max([o[2] for o in A[k]['overlaps']], default=0):.1f}")
    row("total overlap area (px^2)", lambda k: f"{sum(o[2] for o in A[k]['overlaps']):.1f}")
    row("overlap pairs touching 2 different rings", lambda k: f"{X[k]['cross_ring_overlaps']}")

    print(f"\n{'='*70}\nROTATION FIELD\n{'='*70}")
    for k in (L2, L3):
        d = np.array([e[2] for e in A[k]["rot_edges"]])
        print(f"  {k:<4} max={d.max():7.2f}  mean={d.mean():6.2f}  median={np.median(d):6.2f}  "
              f"edges>{BIG_JUMP_DEG:.0f}: {int((d > BIG_JUMP_DEG).sum()):3d}  edges>90: {int((d > 90).sum()):3d}")

    print(f"\n{'='*70}\nEDGE-LENGTH RATIO (deformed/rest), ALL edges\n{'='*70}")
    pct = [0, 5, 25, 50, 75, 95, 100]
    print(f"  percentile   " + "  ".join(f"{p:>6}" for p in pct))
    for k in (L2, L3):
        print(f"  {k:<10}   " + "  ".join(f"{v:6.3f}" for v in np.percentile(A[k]["ratios"], pct))
              + f"   mean={A[k]['ratios'].mean():.3f}")

    print(f"\n{'='*70}\nTHE KEY QUESTION: the UNCONSTRAINED REGION (edges >= 3 hops from every handle)\n{'='*70}")
    for k, s in ((L2, 1.0), (L3, s_final)):
        fr = far_ratios(rest, def_r2 if k == L2 else def_s3, adj, hh)
        print(f"  {k}: n={len(fr)}  edge-ratio percentiles P5/P25/P50/P75/P95 = "
              + " / ".join(f"{v:.3f}" for v in np.percentile(fr, [5, 25, 50, 75, 95]))
              + f"   mean={fr.mean():.3f}   (s used = {s:.3f}; median/s = {np.median(fr) / s:.3f})")
    print(f"  Under the S3 energy an unconstrained edge that follows the objective sits near ratio = s; "
          f"median/s close to 1 means the far field is s-consistent.")

    print(f"\n  Extreme edges (H=handle, R=ring member, O=outside ring; hh = hops to nearest handle):")
    for k in (L2, L3):
        lab = X[k]["lab"]
        print(f"  {k} — 5 most COMPRESSED:")
        for r, i, j in X[k]["smallest"]:
            print(f"     ratio={r:6.3f}  edge({i:3d},{j:3d}) {lab(i)}-{lab(j)}  hh=({hh[i]},{hh[j]})")
        print(f"  {k} — 5 most STRETCHED:")
        for r, i, j in X[k]["largest"]:
            print(f"     ratio={r:6.3f}  edge({i:3d},{j:3d}) {lab(i)}-{lab(j)}  hh=({hh[i]},{hh[j]})")

    print(f"\n{'='*70}\nRING DEMAND vs THE SINGLE s\n{'='*70}")
    print(f"  {'ring':<22}{'pair scale':>11}{'pair/s':>9}{'verdict':>24}{'E_ring R2':>12}{'E_ring S3':>12}")
    conflicts = []
    for h in sorted(rings):
        ps = scale_ua if "underarm" in name_of[h] else scale_sh
        ratio = ps / s_final
        conflict = abs(ratio - 1.0) > 0.15
        if conflict:
            conflicts.append(h)
        e2 = ring_energy(rest, def_r2, rot_r2, 1.0, adj, rings[h])
        e3 = ring_energy(rest, def_s3, rot_s3, s_final, adj, rings[h])
        print(f"  {h:3d} {name_of[h]:<17}{ps:11.3f}{ratio:9.3f}{('CONFLICTS with s' if conflict else 'consistent with s'):>24}{e2:12.1f}{e3:12.1f}")

    print(f"\n{'='*70}\nBOUNDING BOX, PLACEMENT AND HEAD-REGION CHECK\n{'='*70}")
    print(f"  {'':<50} " + "  ".join(f"{k:>14}" for k in (L2, L3)))
    row("deformed/rest width ratio (info only)", lambda k: f"{A[k]['bbox']['w_ratio']:.3f}")
    row("deformed/rest height ratio (info only)", lambda k: f"{A[k]['bbox']['h_ratio']:.3f}")
    row("robust width ratio  [rule]", lambda k: f"{RB[k][0]:.3f}")
    row("robust height ratio [rule]", lambda k: f"{RB[k][1]:.3f}")
    row("far-field mean edge ratio [rule]", lambda k: f"{X[k]['far_mean']:.3f}")
    row("garment top y (canvas px)", lambda k: f"{A[k]['bbox']['top_y']:.1f}")
    row("garment top above avatar shoulder line (px)", lambda k: f"{A[k]['bbox']['above_shoulder']:.1f}")
    row(f"overflow past canvas right edge x={canvas_w} (px)", lambda k: f"{A[k]['bbox']['right_overflow']:.1f}")

    print(f"\n  Diagnostic points: rest | T_underarm baseline | T_shoulder | R2 | S3   (and S3/R2 distance to the T_underarm point)")
    ys, xs = np.where(garment.mask)
    bottom_y = ys.max()
    bxs = xs[ys == bottom_y]
    pts = {"hem (right corner)": (float(bxs.max()), float(bottom_y)),
           "hem (left corner)": (float(bxs.min()), float(bottom_y)),
           "left_sleeve_end": (float(xs.min()), float(ys[xs == xs.min()].mean())),
           "right_sleeve_end": (float(xs.max()), float(ys[xs == xs.max()].mean())),
           "collar top": (float(rest[np.argmin(rest[:, 1])][0]), float(rest[:, 1].min()))}
    seen = set()
    for nm, pt in pts.items():
        v = int(np.argmin(np.linalg.norm(rest - np.array(pt), axis=1)))
        if v in seen:
            continue
        seen.add(v)
        pua, psh = apply_T(T_ua, rest[v]), apply_T(T_sh, rest[v])
        d2 = float(np.linalg.norm(def_r2[v] - pua))
        d3 = float(np.linalg.norm(def_s3[v] - pua))
        print(f"    {nm:19s} v{v:3d}: ({rest[v][0]:.0f},{rest[v][1]:.0f}) | ({pua[0]:.0f},{pua[1]:.0f}) | ({psh[0]:.0f},{psh[1]:.0f})"
              f" | ({def_r2[v][0]:.0f},{def_r2[v][1]:.0f}) | ({def_s3[v][0]:.0f},{def_s3[v][1]:.0f})   dist-to-T_ua: R2={d2:.0f}px S3={d3:.0f}px")

    print(f"\n{'='*70}\nWHERE THE PATHOLOGY LIVES — hops from handles (0=handle, 1=ring member, 2, 3+)\n{'='*70}")
    for k in (L2, L3):
        big = [e for e in A[k]["rot_edges"] if e[2] > BIG_JUMP_DEG]
        print(f"  {k:<4} overlap pairs hh={hop_hist([o[4] for o in A[k]['overlaps']])}   "
              f"inverted tris hh={hop_hist(A[k]['inv_hops_h'])}   rot jumps>{BIG_JUMP_DEG:.0f} hh={hop_hist([e[4] for e in big])}")

    # ---- registered outcome ----
    print(f"\n{'='*70}\nREGISTERED OUTCOME RULE (MID = {MID:.4f})\n{'='*70}")
    w2, h2 = RB[L2]; f2 = X[L2]["far_mean"]
    w3, h3 = RB[L3]; f3 = X[L3]["far_mean"]
    for k, (w, h_, f_) in ((L2, (w2, h2, f2)), (L3, (w3, h3, f3))):
        print(f"  {k:<4} robust-width={w:.3f}  robust-height={h_:.3f}  far-field={f_:.3f}")
    below = [nm for nm, v in (("width", w3), ("height", h3), ("far-field", f3)) if v < MID]
    delta = f3 - f2
    if len(below) == 3:
        outcome = "CONTRACTION"
    elif not below and abs(delta) < 0.05:
        outcome = "NO CHANGE"
    elif not below:
        outcome = f"PARTIAL (far field moved by {delta:+.3f} but nothing is below MID)"
    else:
        outcome = f"PARTIAL / MIXED (below MID: {', '.join(below)}; far-field change vs R2 = {delta:+.3f})"
    print(f"\n  S3 OUTCOME: {outcome}")
    print(f"  far-field mean edge ratio: R2 {f2:.3f} -> S3 {f3:.3f}  (change {delta:+.3f})")
    print(f"  final s = {s_final:.4f}: {s_band}")
    print(f"  rings in conflict with the single s (>15% off): "
          + (", ".join(f"{h} ({name_of[h]})" for h in conflicts) if conflicts else "none"))

    # ---- the eight checks, same on every garment ----
    d2 = np.array([e[2] for e in A[L2]["rot_edges"]])
    d3 = np.array([e[2] for e in A[L3]["rot_edges"]])
    fr2 = far_ratios(rest, def_r2, adj, hh)
    fr3 = far_ratios(rest, def_s3, adj, hh)
    print(f"\n{'='*70}\nEIGHT CHECKS (identical on every garment)  garment={gid}\n{'='*70}")
    print(f"  {'':<52} {'R2':>16} {'S3':>16}")
    def eight(label, a, b):
        print(f"  {label:<52} {a:>16} {b:>16}")
    eight("1 far-field mean edge ratio (3+ hops)", f"{fr2.mean():.3f}", f"{fr3.mean():.3f}")
    eight("1 far-field median edge ratio", f"{np.median(fr2):.3f}", f"{np.median(fr3):.3f}")
    eight("2 robust width ratio", f"{RB[L2][0]:.3f}", f"{RB[L3][0]:.3f}")
    eight("2 robust height ratio", f"{RB[L2][1]:.3f}", f"{RB[L3][1]:.3f}")
    eight("3 max constraint residual (px)", f"{A[L2]['residual']:.6f}", f"{A[L3]['residual']:.6f}")
    eight("4 overlap pairs", f"{len(A[L2]['overlaps'])}", f"{len(A[L3]['overlaps'])}")
    eight("4 max / total overlap area (px^2)",
          f"{max([o[2] for o in A[L2]['overlaps']], default=0):.1f} / {sum(o[2] for o in A[L2]['overlaps']):.1f}",
          f"{max([o[2] for o in A[L3]['overlaps']], default=0):.1f} / {sum(o[2] for o in A[L3]['overlaps']):.1f}")
    eight("5 inverted triangles", f"{len(A[L2]['inverted'])}", f"{len(A[L3]['inverted'])}")
    eight("6 max neighbor rotation jump (deg)", f"{d2.max():.1f}", f"{d3.max():.1f}")
    eight("7 garment top above avatar shoulder line (px)", f"{A[L2]['bbox']['above_shoulder']:.1f}", f"{A[L3]['bbox']['above_shoulder']:.1f}")
    eight("8 global scale s", "1 (fixed)", f"{s_final:.4f}")
    print(f"  (pair scales for this garment: underarm {scale_ua:.4f}, shoulder {scale_sh:.4f}; MID {MID:.4f})")

    label, failing = classify_generalization(
        contraction=(len(below) == 3), far_ok=(f3 < MID),
        res_ok=(A[L3]["residual"] < 1e-6),
        ov_ok=(len(A[L3]["overlaps"]) <= len(A[L2]["overlaps"])),
        inv_ok=(len(A[L3]["inverted"]) <= len(A[L2]["inverted"])),
        rot_ok=(float(d3.max()) <= float(d2.max())),
        integrity_ok=(increases == 0))
    print(f"\n{'='*70}\nGENERALIZATION CLASSIFICATION (registered rule)\n{'='*70}")
    print(f"  garment {gid}  ->  {label}")
    for f in failing:
        print(f"    - {f}")

    # ---- panel ----
    print("\nRendering panel (S3 render | S3 wireframe + T_underarm baseline outline | R2 wireframe)...")
    garment_arr = np.array(garment.image.convert("RGBA"))
    avatar_rgba = np.array(avatar.image.convert("RGBA"))
    layer = render_mesh(rest, def_s3, triangles, garment_arr, canvas_w, canvas_h)
    comp = avatar_rgba.copy()
    alpha = layer[:, :, 3:4].astype(float) / 255.0
    comp[:, :, :3] = (avatar_rgba[:, :, :3] * (1 - alpha) + layer[:, :, :3] * alpha).astype(np.uint8)
    left = Image.fromarray(comp).convert("RGB")
    dr = ImageDraw.Draw(left)
    for t in handle_targets.values():
        dr.ellipse([t[0] - 5, t[1] - 5, t[0] + 5, t[1] + 5], outline=(255, 255, 0), width=2)
    dr.text((6, 6), f"S3 render (s={s_final:.3f})", fill=(255, 255, 255))
    base_poly = [tuple(apply_T(T_ua, rest[i])) for i in range(n_boundary)]
    mid_img = draw_wire(def_s3, triangles, ring_set, anat_handles, canvas_w, canvas_h, shoulder_y, "S3 wireframe")
    ImageDraw.Draw(mid_img).polygon(base_poly, outline=(255, 0, 255))
    right_img = draw_wire(def_r2, triangles, ring_set, anat_handles, canvas_w, canvas_h, shoulder_y, "R2 wireframe")
    ImageDraw.Draw(right_img).polygon(base_poly, outline=(255, 0, 255))
    panel = Image.new("RGB", (canvas_w * 3, canvas_h), (0, 0, 0))
    panel.paste(left, (0, 0)); panel.paste(mid_img, (canvas_w, 0)); panel.paste(right_img, (canvas_w * 2, 0))
    out_path = (f"{OUTPUT_DIR}/checkpoint5_S3_panel_{gid[:8]}.png" if override
                else f"{OUTPUT_DIR}/checkpoint5_S3_panel.png")
    panel.save(out_path)
    print(f"Saved: {out_path}")
    print("  Wireframes: red = canvas, green = mesh, orange = ring vertices, yellow = handles, cyan = avatar shoulder line,")
    print("  magenta outline = where the T_underarm similarity baseline puts the garment boundary (reference only).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="S3 on the R2 rings; optionally override the garment.")
    ap.add_argument("--garment-id", default=None, help="garment UUID to test instead of the golden garment")
    args = ap.parse_args()
    main(args.garment_id)