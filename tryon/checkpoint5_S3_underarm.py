"""
tryon/checkpoint5_S3_underarm.py

S3-UNDERARM-ONLY — scale-aware ARAP (S3: one global scale DOF) with EXACTLY TWO hard handles,
the left and right underarm, and NO shoulder handles / shoulder-derived constraints.

PRIMARY test garment: the real white T-shirt (shoulders: false, 511x447)
    7e16734e-6cb2-4254-adf5-4695549ad5db
CONTROLS (shoulder-present, already characterized): the golden garment and 831098a3-...
    On the controls the SAME mesh is used as in the earlier S3-point runs; only the shoulder handles are
    removed (a pure ablation), so the two-handle result can be compared with the recorded 4-handle one.

KEY STRUCTURAL FACT (found while building this, before any data): two point handles fix a similarity transform
exactly (4 equations; scale, angle, 2 translations). That similarity T' (the one mapping the two handle VERTICES to
their targets) therefore has exactly ZERO S3 energy and is the unique global minimum. S3-underarm converges to a
rigid+scale copy of the garment. See the registered text below for what that does to the experiment.

Frozen / imported, unchanged: avatar, mesh generation, adjacency, uniform weights, cold start, S3 energy
and solver, iteration cap, convergence rule, overlap detector, analysis, classification function.

Two small, flagged accommodations for a shoulderless garment (neither changes the formulation):
  1. Mesh: the frozen mesh builder reads shoulder landmarks. For a shoulderless garment it is given DUMMY
     shoulder landmarks that coincide with boundary vertex 0, so they match an existing boundary vertex and add
     no Steiner point. The mesh is exactly boundary + underarm handling + Poisson sampling.
  2. Builder: build_garment_representation() calls shoulder detection with no try/except, while the asset
     builder that reported shoulders=false degrades silently. For the primary garment ONLY, an exception in
     shoulder detection is treated as "shoulders unavailable" (the same tolerance), and this is printed.

Run: python -m tryon.checkpoint5_S3_underarm [--garment-id ID] [--controls ID ...] [--no-controls]
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image, ImageDraw

import tryon.garment_representation as garment_module
from tryon.avatar_representation import build_avatar_representation
from tryon.piecewise_continuous_test import solve_similarity_2point
from tryon.checkpoint5_ARAP0_deformation_field import (
    build_dense_mesh, build_adjacency, run_arap, local_step, trace_and_simplify,
    GOLDEN_GARMENT_ID, AVATAR_USER_ID, MAX_ITERATIONS,
)
from tryon.checkpoint5_ARAP1 import (
    get_anatomical_handles, analyze, hop_hist, multi_bfs, render_mesh, BIG_JUMP_DEG,
)
from tryon.checkpoint5_ARAP1_R2 import pair_by_x, robust_ratio, apply_T, draw_wire
from tryon.checkpoint5_S3 import (
    run_arap_scaled, arap_energy, directed_edges, far_ratios, classify_generalization, s_star,
)

OUTPUT_DIR = "outputs"
WHITE_TEE_ID = "7e16734e-6cb2-4254-adf5-4695549ad5db"
SECOND_GARMENT_ID = "831098a3-8778-480f-9a11-aa9df826d0f2"

# Recorded real S3-point (4 handle) results from the previous run; drift guard for the controls.
KNOWN_4H = {
    GOLDEN_GARMENT_ID: dict(s=0.36869, E=1860.5798),
    SECOND_GARMENT_ID: dict(s=0.36923, E=3986.4289),
}

OUT_SUFFICIENT = "TWO UNDERARM HANDLES SUFFICE — shoulders are optional (primary passes; controls not degraded)"
OUT_HELP = "SHOULDERS HELP WHEN AVAILABLE — primary passes without them, but removing shoulders degrades a shoulder-present control"
OUT_SPECIFIC = "GARMENT-SPECIFIC — two handles suffice on the controls but the primary garment does not pass"
OUT_NEEDED = "SHOULDERS APPEAR TO STABILIZE THE FORMULATION — primary does not pass and removing shoulders also degrades a control"
OUT_NOCTRL_PASS = "PASS (no controls run) — cannot separate 'optional' from 'helpful'"
OUT_NOCTRL_FAIL = "NOT PASS (no controls run) — cannot separate garment-specific from shoulder-required"

EXT_ITERS = 1000

REGISTERED = """
======================================================================
REGISTERED BEFORE RUNNING (S3-underarm-only) — REVISED before any data existed
======================================================================
REVISION NOTICE. The first draft registered a four-way outcome (primary PASS/not x controls OK/not) meant to say
whether shoulders are 'required' or 'optional'. While building it I found that with exactly two point handles the
S3 energy has an exact zero-energy solution: the similarity T' through the two handle vertices. S3-underarm must
therefore converge to a rigid+scale copy of the garment, for ANY garment. Far-field contraction, zero overlaps,
zero inversions and constant rotation are then guaranteed by construction, so the draft's PASS / 'as clean as 4
handles' outcome cannot discriminate anything. It is still PRINTED (clearly labelled) but is NOT the registered
outcome. Verified in a sandbox first: the exact similarity has S3 energy ~1e-25, and the iteration reaches s = |T'|
and E = 0 by about 200 iterations; at the standard 60 it is only partly converged.

What this experiment CAN establish (registered):
  C1  CONVERGENCE TO THE SIMILARITY (mechanical, per garment). Re-run S3-underarm for 1000 iterations. CONFIRMED if
      E_1000 < 1e-3, |s_1000 - |T'|| < 1e-4 and max_i ||p'_i - T'(p_i)|| < 0.05 px. A failure on a real garment would
      be a surprise and means something about the real mesh is not what the theory assumes. Also reported at 60 it.
  C2  WHAT THE SHOULDER HANDLES BUY (controls only; informational, judged visually): the similarity placement vs the
      recorded 4-handle placement: renders side by side, garment-shoulder-landmark -> avatar-shoulder distances
      (the 4-handle result pins them to 0), collar / hem / sleeve-end positions.
  C3  PLACEMENT OF THE SHOULDERLESS GARMENT (informational, judged visually): collar-top above the avatar shoulder
      line, overflow, hem, sleeve-end positions, and the render.
No pass/fail is registered for C2 or C3: whether a rigid underarm-similarity placement is an acceptable FIT at the
shoulders is a fit-quality question these mechanical metrics do not answer.
Consequence if C1 holds (expected): for shoulderless garments local ARAP deformation adds nothing over the global
similarity T' (the same family as the production lower-zone transform); the capability-dependent design is
'>= 3 independent constraints -> S3 ARAP, 2 -> closed-form similarity'.
Printed for traceability only (NON-INFORMATIVE BY CONSTRUCTION): the draft's classification vs ARAP-0b, 'as clean as
4 handles' on controls, and the draft's four-way label.
Upper-region indicators (rest y above the underarm handles' mean y) are printed as before; for the similarity they are
trivially clean.
"""


# ----------------------------------------------------------------------------- helpers
class _ShoulderlessShim:
    """Gives the frozen mesh builder dummy shoulder landmarks that coincide with boundary vertex 0."""
    def __init__(self, g, dummy):
        self.mask, self.image = g.mask, g.image
        self.left_underarm, self.right_underarm = g.left_underarm, g.right_underarm
        self.left_shoulder = dummy
        self.right_shoulder = dummy


class tolerant_shoulders:
    """For the primary garment only: an exception inside shoulder detection means 'shoulders unavailable'."""
    def __enter__(self):
        self.orig = getattr(garment_module, "get_garment_shoulder_points", None)
        if self.orig is not None:
            orig = self.orig
            def wrapped(*a, **k):
                try:
                    return orig(*a, **k)
                except Exception as exc:
                    print(f"  NOTE: shoulder detection raised {type(exc).__name__}: {exc}\n"
                          f"        -> treated as shoulders unavailable (the same tolerance the asset builder applies)")
                    return None, None
            garment_module.get_garment_shoulder_points = wrapped
        return self
    def __exit__(self, *exc):
        if self.orig is not None:
            garment_module.get_garment_shoulder_points = self.orig


def underarm_handles(garment, avatar, landmark_index):
    a_ua = pair_by_x(avatar.left_underarm, avatar.right_underarm)
    left_first = garment.left_underarm[0] < garment.right_underarm[0]
    pairs = [("left_underarm" if left_first else "right_underarm", a_ua[0]),
             ("right_underarm" if left_first else "left_underarm", a_ua[1])]
    return {landmark_index[nm]: np.array(a_pt, dtype=float) for nm, a_pt in pairs}


def final_outcome(primary_label, controls):
    w_pass = primary_label == "PASS"
    if not controls:
        return OUT_NOCTRL_PASS if w_pass else OUT_NOCTRL_FAIL
    c_ok = all(c["label"] == "PASS" and c["as_clean_vs_4h"] is True for c in controls)
    if w_pass:
        return OUT_SUFFICIENT if c_ok else OUT_HELP
    return OUT_SPECIFIC if c_ok else OUT_NEEDED


def upper_indicators(rest, deformed, A, s, triangles, adj, y_cut):
    up = rest[:, 1] < y_cut
    rot_up = [e[2] for e in A["rot_edges"] if up[e[0]] and up[e[1]]]
    ratios_up = []
    for i in adj:
        for j in adj[i]:
            if j > i and up[i] and up[j]:
                rl = float(np.linalg.norm(rest[i] - rest[j]))
                if rl > 1e-9:
                    ratios_up.append(float(np.linalg.norm(deformed[i] - deformed[j])) / rl)
    cent_up = [bool(rest[list(t)][:, 1].mean() < y_cut) for t in triangles]
    share = float(np.mean(np.abs(np.array(ratios_up) / s - 1.0) > 0.35)) if ratios_up else float("nan")
    return dict(n_tri=sum(cent_up), n_edges=len(rot_up),
                rot_max=float(max(rot_up)) if rot_up else float("nan"),
                rot_gt45=int(sum(1 for r in rot_up if r > BIG_JUMP_DEG)),
                share=share,
                ov=sum(1 for o in A["overlaps"] if cent_up[o[0]] or cent_up[o[1]]),
                inv=sum(1 for t in A["inverted"] if cent_up[t]))


# ----------------------------------------------------------------------------- one garment
def run_one(gid, avatar, role):
    bar = "=" * 70
    print(f"\n{bar}\n{role.upper()} GARMENT {gid}\n{bar}")
    try:
        if role == "primary":
            with tolerant_shoulders():
                garment = garment_module.build_garment_representation(gid, AVATAR_USER_ID)
        else:
            garment = garment_module.build_garment_representation(gid, AVATAR_USER_ID)
    except Exception as exc:
        print(f"NOT TESTABLE: garment representation failed: {type(exc).__name__}: {exc}")
        return None

    has_sh = bool(garment.shoulders_available)
    print(f"  garment image size={garment.image.size}  underarms L={tuple(garment.left_underarm)} R={tuple(garment.right_underarm)}"
          f"  shoulders_available={has_sh}")
    if role == "control" and not has_sh:
        print("NOT TESTABLE as a control: this control garment has no shoulders (nothing to ablate).")
        return None
    if role == "primary" and has_sh:
        print("  NOTE: the primary garment HAS shoulders on this machine; its mesh is built with them (control-style), so this run is an")
        print("        ablation, not the shoulderless contract.")

    try:
        if has_sh:
            rest, triangles, landmark_index, n_boundary = build_dense_mesh(garment)
            mesh_mode = "built with the garment's real shoulder landmarks (as in the earlier S3 runs)"
        else:
            boundary0 = trace_and_simplify(garment.mask, epsilon=2.0)
            shim = _ShoulderlessShim(garment, tuple(boundary0[0]))
            rest, triangles, landmark_index, n_boundary = build_dense_mesh(shim)
            assert landmark_index["left_shoulder"] == 0 and landmark_index["right_shoulder"] == 0, "dummy shoulders did not match boundary vertex 0"
            mesh_mode = "shoulderless: dummy shoulder landmarks matched boundary vertex 0 (no Steiner points added for them)"
    except Exception as exc:
        print(f"NOT TESTABLE: mesh construction failed: {type(exc).__name__}: {exc}")
        return None
    n = len(rest)
    adj = build_adjacency(n, triangles)
    targets2 = underarm_handles(garment, avatar, landmark_index)
    anat2 = set(targets2)
    ua_unmatched = sorted(v for v in anat2 if v >= n_boundary)
    print(f"  mesh: {n} vertices ({n_boundary} boundary), {len(triangles)} triangles; {mesh_mode}")
    print(f"  underarm handles {sorted(anat2)}: " + ("both matched boundary vertices" if not ua_unmatched else f"vertices {ua_unmatched} are interior Steiner points (nudged landmark)"))
    canvas_w, canvas_h = avatar.image.size
    shoulder_y = float(min(avatar.left_shoulder[1], avatar.right_shoulder[1]))
    hh2 = multi_bfs(adj, anat2, n)

    g_ua = pair_by_x(garment.left_underarm, garment.right_underarm)
    a_ua = pair_by_x(avatar.left_underarm, avatar.right_underarm)
    T_ua = np.array(solve_similarity_2point(list(g_ua), list(a_ua)))
    scale_ua = float(np.hypot(T_ua[0, 0], T_ua[1, 0]))
    rot_ua = float(np.degrees(np.arctan2(T_ua[1, 0], T_ua[0, 0])))
    MID = (1.0 + scale_ua) / 2.0
    print(f"  T_underarm: scale={scale_ua:.4f} rotation={rot_ua:+.2f} deg   MID={MID:.4f}   (the only required scale: {scale_ua:.4f})")
    y_cut = float(np.mean([rest[v][1] for v in anat2]))

    # controls: 4-handle result on the SAME mesh (reference), plus subset check of the handle construction
    targets4 = anat4 = None
    if has_sh:
        targets4 = get_anatomical_handles(garment, avatar, landmark_index)
        anat4 = set(targets4)
        same_sub = all(v in targets4 and np.allclose(targets4[v], targets2[v]) for v in anat2)
        print(f"  handle-construction self-check: the two underarm targets equal the 4-handle function's underarm targets -> {'PASS' if same_sub else 'FAIL'}")

    # ---------------- solves
    print(f"\nRunning ARAP-0b (2 underarm handles, s fixed at 1, {MAX_ITERATIONS} it)...")
    def0, rot0 = run_arap(rest, adj, anat2, targets2, MAX_ITERATIONS)
    chk, _, _, _, _ = run_arap_scaled(rest, adj, anat2, targets2, MAX_ITERATIONS, freeze_s=True)
    max_diff = float(np.abs(chk - def0).max())
    print(f"Integrity check 1: S3 solver with s frozen at 1 vs the frozen solver (2 handles): max |diff| = {max_diff:.3e} px  ({'PASS' if max_diff < 1e-6 else 'FAIL'})")
    if max_diff >= 1e-6:
        print("  Stopping: results would not be attributable to s alone.")
        sys.exit(1)
    print("Running S3-underarm (2 handles, s free)...")
    defu, rotu, s_u, trace, clamp_hits = run_arap_scaled(rest, adj, anat2, targets2, MAX_ITERATIONS)
    def4 = rot4 = s_4 = None
    if has_sh:
        print("Running S3-point (4 handles, recorded reference on the same mesh)...")
        def4, rot4, s_4, _, _ = run_arap_scaled(rest, adj, anat4, targets4, MAX_ITERATIONS)
    I, J = directed_edges(adj)
    E0 = arap_energy(rest, def0, rot0, 1.0, I, J)
    Eu = arap_energy(rest, defu, rotu, s_u, I, J)
    E4 = arap_energy(rest, def4, rot4, s_4, I, J) if has_sh else None
    print(f"  this run: ARAP-0b E(s=1)={E0:.4f} | S3-underarm s={s_u:.5f} E={Eu:.4f}" + (f" | S3-point(4h) s={s_4:.5f} E={E4:.4f}" if has_sh else ""))
    kn = KNOWN_4H.get(gid)
    if has_sh and kn:
        ok = abs(s_4 - kn["s"]) < 6e-6 and abs(E4 - kn["E"]) < 2e-6 * kn["E"] + 1e-3
        print(f"  drift guard, S3-point(4h) vs recorded s={kn['s']:.5f} E={kn['E']:.4f}  -> {'MATCH' if ok else 'MISMATCH — the 4-handle reference differs from the recorded S3-point result'}")

    # ---------------- exact similarity through the two handle vertices, and the extended run (C1)
    hs_sorted = sorted(anat2, key=lambda v: rest[v][0])
    Tex = np.array(solve_similarity_2point([tuple(rest[hs_sorted[0]]), tuple(rest[hs_sorted[1]])],
                                            [tuple(targets2[hs_sorted[0]]), tuple(targets2[hs_sorted[1]])]))
    scale_ex = float(np.hypot(Tex[0, 0], Tex[1, 0]))
    exact = np.array([apply_T(Tex, p) for p in rest])
    R_ex = np.repeat((Tex[:, :2] / scale_ex)[None, :, :], n, axis=0)
    E_exact = arap_energy(rest, exact, R_ex, scale_ex, I, J)
    dev60 = float(np.linalg.norm(defu - exact, axis=1).max())
    print(f"\n{'-'*70}\nC1  CONVERGENCE TO THE EXACT SIMILARITY T' (through the two handle vertices)\n{'-'*70}")
    print(f"  |T'| = {scale_ex:.5f} (T_underarm from the raw landmarks: {scale_ua:.5f}; they differ by the mesh-vertex-vs-landmark offset)")
    print(f"  S3 energy of the exact similarity configuration: {E_exact:.3e}   (a zero-energy solution exists)")
    print(f"  at {MAX_ITERATIONS} iterations: s={s_u:.5f} (s-|T'|={s_u - scale_ex:+.5f})  E={Eu:.4f}  max ||p'-T'(p)||={dev60:.3f}px")
    print(f"  re-running {EXT_ITERS} iterations...")
    defx, rotx, s_x, _, _ = run_arap_scaled(rest, adj, anat2, targets2, EXT_ITERS)
    E_x = arap_energy(rest, defx, rotx, s_x, I, J)
    dev_x = float(np.linalg.norm(defx - exact, axis=1).max())
    c1 = bool(E_x < 1e-3 and abs(s_x - scale_ex) < 1e-4 and dev_x < 0.05)
    print(f"  at {EXT_ITERS} iterations: s={s_x:.5f} (s-|T'|={s_x - scale_ex:+.5f})  E={E_x:.6f}  max ||p'-T'(p)||={dev_x:.4f}px")
    print(f"  C1: {'CONFIRMED' if c1 else 'NOT CONFIRMED — investigate (the real mesh does not behave as the theory assumes)'}")

    # ---------------- convergence
    print(f"\n{'-'*70}\nS3-UNDERARM CONVERGENCE (iterations 1-5, then every 5th)\n{'-'*70}")
    prev, increases = None, 0
    first_below = {1e-3: None, 1e-4: None, 1e-5: None}
    for it, E, s, md in trace:
        rel = abs(E - prev) / max(prev, 1e-12) if prev is not None else float("nan")
        if prev is not None:
            if E > prev * (1 + 1e-9):
                increases += 1
            for t in first_below:
                if first_below[t] is None and rel < t:
                    first_below[t] = it
        if it <= 5 or it % 5 == 0:
            print(f"  iter {it:3d}  E={E:14.4f}  dE_rel={rel if it > 1 else float('nan'):.6f}  s={s:.5f}  max_disp={md:.4f}px")
        prev = E
    R_fin = local_step(rest, defu, adj, n)
    s_chk = s_star(rest, defu, R_fin, I, J)
    print(f"  Integrity check 2: iterations where E INCREASED = {increases}   ({'PASS' if increases == 0 else 'FAIL'})   |   s clamp activations = {clamp_hits}")
    print(f"  Integrity check 3: final s = {s_u:.5f}; re-solved on the final configuration = {s_chk:.5f}  (relative gap {abs(s_chk - s_u) / s_u:.2e})")
    print(f"  first iteration with dE_rel < 1e-3 / 1e-4 / 1e-5: {first_below[1e-3]} / {first_below[1e-4]} / {first_below[1e-5]}")
    rel_s = s_u / scale_ua
    print(f"  final s = {s_u:.4f} = {rel_s:.3f} x the underarm scale ({scale_ua:.4f})")

    # ---------------- analysis
    A0 = analyze(rest, def0, rot0, triangles, adj, anat2, anat2, targets2, canvas_w, shoulder_y)
    Au = analyze(rest, defu, rotu, triangles, adj, anat2, anat2, targets2, canvas_w, shoulder_y)
    A4 = analyze(rest, def4, rot4, triangles, adj, anat4, anat4, targets4, canvas_w, shoulder_y) if has_sh else None
    cols = [("ARAP-0b", def0, A0, 1.0, rot0), ("S3-underarm", defu, Au, s_u, rotu)]
    if has_sh:
        cols.append(("S3-point(4h)", def4, A4, s_4, rot4))
    names = [c[0] for c in cols]
    CA = {c[0]: c[2] for c in cols}
    RB = {c[0]: (robust_ratio(rest, c[1], 0), robust_ratio(rest, c[1], 1)) for c in cols}
    FR = {c[0]: far_ratios(rest, c[1], adj, hh2) for c in cols}
    ROT = {c[0]: np.array([e[2] for e in c[2]["rot_edges"]]) for c in cols}
    S = {c[0]: c[3] for c in cols}

    def row(label, f, w=15):
        print(f"  {label:<48}" + "".join(f"{f(k):>{w}}" for k in names))

    print(f"\n{'='*70}\nEIGHT CHECKS  garment={gid}\n{'='*70}")
    print(f"  {'':<48}" + "".join(f"{k:>15}" for k in names))
    row("1 far-field mean edge ratio (3+ hops from the 2 handles)", lambda k: f"{FR[k].mean():.3f}")
    row("1 far-field median edge ratio", lambda k: f"{np.median(FR[k]):.3f}")
    row("2 robust width ratio", lambda k: f"{RB[k][0]:.3f}")
    row("2 robust height ratio", lambda k: f"{RB[k][1]:.3f}")
    row("3 max constraint residual (px)", lambda k: f"{CA[k]['residual']:.6f}")
    row("4 overlap pairs", lambda k: f"{len(CA[k]['overlaps'])}")
    row("4 max / total overlap area (px^2)", lambda k: f"{max([o[2] for o in CA[k]['overlaps']], default=0):.1f}/{sum(o[2] for o in CA[k]['overlaps']):.1f}")
    row("5 inverted triangles", lambda k: f"{len(CA[k]['inverted'])}")
    row("6 max neighbor rotation jump (deg)", lambda k: f"{ROT[k].max():.1f}")
    row("6 rotation jumps > 45 deg", lambda k: f"{int((ROT[k] > BIG_JUMP_DEG).sum())}")
    row("7 garment top above avatar shoulder line (px)", lambda k: f"{CA[k]['bbox']['above_shoulder']:.1f}")
    row("7 overflow past canvas right edge (px)", lambda k: f"{CA[k]['bbox']['right_overflow']:.1f}")
    row("8 global scale s", lambda k: ("1 (fixed)" if k == "ARAP-0b" else f"{S[k]:.4f}"))
    if has_sh:
        hh4 = multi_bfs(adj, anat4, n)
        print(f"  common far-field set (>=3 hops from all FOUR handles), mean edge ratio:  "
              + "  ".join(f"{k}={far_ratios(rest, c[1], adj, hh4).mean():.3f}" for k, c in zip(names, cols)))

    print(f"\n  EDGE-LENGTH RATIO percentiles, all edges (0/5/25/50/75/95/100):")
    for c in cols:
        print(f"    {c[0]:<13}" + "  ".join(f"{v:6.3f}" for v in np.percentile(c[2]["ratios"], [0, 5, 25, 50, 75, 95, 100])) + f"   mean={c[2]['ratios'].mean():.3f}")
    print(f"  FAR-FIELD edge-ratio P5/P25/P50/P75/P95 (median / s):")
    for c in cols:
        print(f"    {c[0]:<13}" + " / ".join(f"{v:.3f}" for v in np.percentile(FR[c[0]], [5, 25, 50, 75, 95])) + f"   ({np.median(FR[c[0]]) / c[3]:.3f})")

    # upper-region indicators
    print(f"\n  UPPER-REGION INDICATORS (rest y < {y_cut:.0f}, the underarm handles' mean y; informational):")
    UI = {c[0]: upper_indicators(rest, c[1], c[2], c[3], triangles, adj, y_cut) for c in cols}
    print(f"    {'':<46}" + "".join(f"{k:>15}" for k in names))
    for label, key, fmt in (("upper-region triangles / edges", None, None), ("max rotation jump (deg)", "rot_max", "{:.1f}"),
                            ("rotation jumps > 45 deg", "rot_gt45", "{}"), ("edges deviating >35% from the run's own s", "share", "{:.1%}"),
                            ("overlap pairs touching the region", "ov", "{}"), ("inverted triangles in the region", "inv", "{}")):
        if key is None:
            print(f"    {label:<46}" + "".join(f"{str(UI[k]['n_tri']) + '/' + str(UI[k]['n_edges']):>15}" for k in names))
        else:
            vals = [("n/a (s=1)" if (key == "share" and k == "ARAP-0b") else fmt.format(UI[k][key])) for k in names]
            print(f"    {label:<46}" + "".join(f"{v:>15}" for v in vals))

    # extreme edges, placement
    print(f"\n  Extreme edges for S3-underarm (H=handle, O=other; hh = hops to the nearest of the 2 handles):")
    edges = []
    for i in adj:
        for j in adj[i]:
            if j > i:
                rl = float(np.linalg.norm(rest[i] - rest[j]))
                if rl > 1e-9:
                    edges.append((float(np.linalg.norm(defu[i] - defu[j])) / rl, i, j, rl))
    edges.sort()
    lab = lambda v: "H" if v in anat2 else "O"
    for tag, rows_ in (("5 most COMPRESSED", edges[:5]), ("5 most STRETCHED", edges[-5:][::-1])):
        print(f"    {tag}:")
        for r, i, j, rl in rows_:
            print(f"      ratio={r:6.3f} edge({i:3d},{j:3d}) {lab(i)}-{lab(j)} hh=({hh2[i]},{hh2[j]}) rest=({rest[i][0]:.0f},{rest[i][1]:.0f})-({rest[j][0]:.0f},{rest[j][1]:.0f}) len={rl:.1f}px")
    print(f"\n  Diagnostic points: rest | T_underarm baseline | " + " | ".join(names[1:]) + "   (distance to the baseline point)")
    ys, xs = np.where(garment.mask)
    bottom_y = ys.max()
    bxs = xs[ys == bottom_y]
    pts = {"hem (right corner)": (float(bxs.max()), float(bottom_y)), "hem (left corner)": (float(bxs.min()), float(bottom_y)),
           "left_sleeve_end": (float(xs.min()), float(ys[xs == xs.min()].mean())),
           "right_sleeve_end": (float(xs.max()), float(ys[xs == xs.max()].mean())),
           "collar top": (float(rest[np.argmin(rest[:, 1])][0]), float(rest[:, 1].min()))}
    seen = set()
    for nm, pt in pts.items():
        v = int(np.argmin(np.linalg.norm(rest - np.array(pt), axis=1)))
        if v in seen:
            continue
        seen.add(v)
        base = apply_T(T_ua, rest[v])
        parts = [f"({c[1][v][0]:.0f},{c[1][v][1]:.0f}) {np.linalg.norm(c[1][v] - base):.0f}px" for c in cols[1:]]
        print(f"    {nm:19s} v{v:3d}: ({rest[v][0]:.0f},{rest[v][1]:.0f}) | ({base[0]:.0f},{base[1]:.0f}) | " + " | ".join(parts))
    print(f"\n  WHERE THE PATHOLOGY LIVES (hops from the 2 handles: 0, 1, 2, 3+):")
    for c in cols:
        big = [e for e in c[2]["rot_edges"] if e[2] > BIG_JUMP_DEG]
        print(f"    {c[0]:<13} overlap pairs hh={hop_hist([o[4] for o in c[2]['overlaps']])}  inverted tris hh={hop_hist(c[2]['inv_hops_h'])}  rot jumps>{BIG_JUMP_DEG:.0f} hh={hop_hist([e[4] for e in big])}")

    # ---------------- registered classification
    wu, hu = RB["S3-underarm"]
    fu = FR["S3-underarm"].mean()
    below = [nm for nm, v in (("width", wu), ("height", hu), ("far-field", fu)) if v < MID]
    label, failing = classify_generalization(
        contraction=(len(below) == 3), far_ok=(fu < MID),
        res_ok=(Au["residual"] < 1e-6),
        ov_ok=(len(Au["overlaps"]) <= len(A0["overlaps"])),
        inv_ok=(len(Au["inverted"]) <= len(A0["inverted"])),
        rot_ok=(float(ROT["S3-underarm"].max()) <= float(ROT["ARAP-0b"].max())),
        integrity_ok=(increases == 0))
    print(f"\n{'='*70}\nDRAFT CLASSIFICATION vs ARAP-0b (NON-INFORMATIVE BY CONSTRUCTION; printed for traceability)  MID={MID:.4f}\n{'='*70}")
    print(f"  S3-underarm: robust-width={wu:.3f} robust-height={hu:.3f} far-field={fu:.3f}  (below MID: {', '.join(below) if below else 'none'})")
    print(f"  garment {gid} -> {label}")
    for f in failing:
        print(f"    - {f}")
    as_clean = None
    if has_sh:
        items = [("overlap pairs", len(Au["overlaps"]), len(A4["overlaps"])),
                 ("inverted triangles", len(Au["inverted"]), len(A4["inverted"])),
                 ("max rotation jump", float(ROT["S3-underarm"].max()), float(ROT["S3-point(4h)"].max())),
                 (f"rotation jumps > {BIG_JUMP_DEG:.0f}", int((ROT["S3-underarm"] > BIG_JUMP_DEG).sum()), int((ROT["S3-point(4h)"] > BIG_JUMP_DEG).sum()))]
        as_clean = all(a <= b for _, a, b in items)
        print(f"  AS CLEAN AS 4 HANDLES (non-informative: a rigid copy cannot overlap or invert)? {'YES' if as_clean else 'NO'}")
        for nm, a, b in items:
            print(f"    {nm:<26} S3-underarm {a:>8.1f}   S3-point(4h) {b:>8.1f}   {'ok' if a <= b else 'WORSE'}")
        g_sh = pair_by_x(garment.left_shoulder, garment.right_shoulder)
        a_sh = pair_by_x(avatar.left_shoulder, avatar.right_shoulder)
        d_sim = [float(np.linalg.norm(apply_T(Tex, g_sh[k]) - np.array(a_sh[k], float))) for k in (0, 1)]
        print(f"  C2 garment SHOULDER landmark -> avatar shoulder distance (small-x / large-x): similarity {d_sim[0]:.1f}px / {d_sim[1]:.1f}px;"
              f" 4-handle result: pinned (0 / 0 px)")
        print(f"  REMOVAL EFFECT of the shoulder handles on this garment (S3-underarm minus S3-point(4h)):")
        print(f"    far-field mean {fu - FR['S3-point(4h)'].mean():+.3f} | robust width {RB['S3-underarm'][0] - RB['S3-point(4h)'][0]:+.3f} | "
              f"robust height {RB['S3-underarm'][1] - RB['S3-point(4h)'][1]:+.3f} | s {s_u - s_4:+.4f} | "
              f"top above shoulder line {Au['bbox']['above_shoulder'] - A4['bbox']['above_shoulder']:+.1f}px")
        print(f"    upper-region: max rotation jump {UI['S3-underarm']['rot_max'] - UI['S3-point(4h)']['rot_max']:+.1f} deg | "
              f"edges deviating >35%: {UI['S3-underarm']['share'] - UI['S3-point(4h)']['share']:+.1%} | "
              f"overlaps {UI['S3-underarm']['ov'] - UI['S3-point(4h)']['ov']:+d} | inverted {UI['S3-underarm']['inv'] - UI['S3-point(4h)']['inv']:+d}")

    # ---------------- panel
    print("\nRendering panel...")
    garment_arr = np.array(garment.image.convert("RGBA"))
    avatar_rgba = np.array(avatar.image.convert("RGBA"))
    def render_on_avatar(defm, handles, title):
        layer = render_mesh(rest, defm, triangles, garment_arr, canvas_w, canvas_h)
        comp = avatar_rgba.copy()
        alpha = layer[:, :, 3:4].astype(float) / 255.0
        comp[:, :, :3] = (avatar_rgba[:, :, :3] * (1 - alpha) + layer[:, :, :3] * alpha).astype(np.uint8)
        im = Image.fromarray(comp).convert("RGB")
        d = ImageDraw.Draw(im)
        for t in handles.values():
            d.ellipse([t[0] - 5, t[1] - 5, t[0] + 5, t[1] + 5], outline=(255, 255, 0), width=2)
        d.text((6, 6), title, fill=(255, 255, 255))
        return im
    base_poly = [tuple(apply_T(T_ua, rest[i])) for i in range(n_boundary)]
    panel = Image.new("RGB", (canvas_w * 3, canvas_h), (0, 0, 0))
    if has_sh:
        panel.paste(render_on_avatar(def4, targets4, f"S3-point(4h) render (s={s_4:.3f})"), (0, 0))
        panel.paste(render_on_avatar(defu, targets2, f"S3-underarm = similarity (s={s_u:.3f})"), (canvas_w, 0))
        ov = draw_wire(def4, triangles, set(), anat4, canvas_w, canvas_h, shoulder_y, "wireframes: green=4h, orange=2h")
        od = ImageDraw.Draw(ov)
        for tri in triangles:
            od.polygon([tuple(defu[i]) for i in tri], outline=(255, 140, 0))
        od.polygon(base_poly, outline=(255, 0, 255))
        panel.paste(ov, (canvas_w * 2, 0))
    else:
        panel.paste(render_on_avatar(defu, targets2, f"S3-underarm render (s={s_u:.3f})"), (0, 0))
        mid_img = draw_wire(defu, triangles, set(), anat2, canvas_w, canvas_h, shoulder_y, "S3-underarm wireframe")
        ImageDraw.Draw(mid_img).polygon(base_poly, outline=(255, 0, 255))
        panel.paste(mid_img, (canvas_w, 0))
        right_img = draw_wire(def0, triangles, set(), anat2, canvas_w, canvas_h, shoulder_y, "ARAP-0b wireframe (s=1)")
        ImageDraw.Draw(right_img).polygon(base_poly, outline=(255, 0, 255))
        panel.paste(right_img, (canvas_w * 2, 0))
    out_path = f"{OUTPUT_DIR}/checkpoint5_S3underarm_panel_{gid[:8]}.png"
    panel.save(out_path)
    print(f"Saved: {out_path}   (red = canvas, green/orange = mesh, yellow = handles, cyan = avatar shoulder line, magenta = T_underarm baseline outline)")

    return dict(gid=gid, role=role, label=label, as_clean_vs_4h=as_clean, s=s_u, far=float(fu), failing=failing, has_shoulders=has_sh,
                c1=c1, scale_ex=scale_ex, s_ext=s_x, E_ext=E_x, dev_ext=dev_x)


def main(primary_id, control_ids):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(__doc__)
    print(REGISTERED)
    print("Building the avatar once...")
    try:
        avatar = build_avatar_representation()
    except Exception as exc:
        print(f"avatar build failed: {type(exc).__name__}: {exc}")
        sys.exit(2)
    primary = run_one(primary_id, avatar, "primary")
    if primary is None:
        print("\nThe primary garment was not testable; no outcome can be registered. (A setup incompatibility, not a result.)")
        sys.exit(2)
    controls = [r for r in (run_one(g, avatar, "control") for g in control_ids) if r is not None]
    bar = "=" * 70
    print(f"\n{bar}\nREGISTERED CHECK C1 (convergence to the exact similarity), all garments\n{bar}")
    for r in [primary] + controls:
        print(f"  {r['role']:<8} {r['gid'][:8]}  |T'|={r['scale_ex']:.5f}  s(1000 it)={r['s_ext']:.5f}  E(1000 it)={r['E_ext']:.2e}  "
              f"max dev={r['dev_ext']:.4f}px   -> {'CONFIRMED' if r['c1'] else 'NOT CONFIRMED'}")
    print(f"\n  C2 / C3 (placement) are informational: judge the saved panels visually.")
    print(f"\n{bar}\nDRAFT FOUR-WAY LABEL — NON-INFORMATIVE BY CONSTRUCTION (do not record as a finding)\n{bar}")
    print(f"  primary {primary['gid'][:8]}: draft class {primary['label']}  s={primary['s']:.4f}  far-field={primary['far']:.3f}")
    for c in controls:
        print(f"  control {c['gid'][:8]}: draft class {c['label']:<8} as clean as 4 handles: {str(c['as_clean_vs_4h']):<5} s={c['s']:.4f}")
    print(f"  draft label: {final_outcome(primary['label'], controls)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="S3 with two underarm handles on the real shoulderless garment, with shoulder-ablation controls.")
    ap.add_argument("--garment-id", default=WHITE_TEE_ID)
    ap.add_argument("--controls", nargs="*", default=[GOLDEN_GARMENT_ID, SECOND_GARMENT_ID])
    ap.add_argument("--no-controls", action="store_true")
    args = ap.parse_args()
    main(args.garment_id, [] if args.no_controls else args.controls)