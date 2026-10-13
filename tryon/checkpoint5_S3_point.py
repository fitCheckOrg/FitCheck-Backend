"""
tryon/checkpoint5_S3_point.py

S3-POINT — scale-aware ARAP (one global scale DOF, S3) with the original FOUR single-vertex
point handles instead of the R2 rings. The missing cell in the experiment matrix:

                        no global scale     global scale (S3)
    point handles       ARAP-0   FAIL       S3-point   <- this script
    1-ring translated   R1       FAIL
    1-ring similarity   R2                  S3-ring    (re-run here as the comparison column)

ONLY CHANGE vs S3-ring: the constrained set is the four handle vertices, pinned at their exact
avatar-landmark targets. NO ring-derived soft constraints of any kind. Everything else is the
frozen/imported code: mesh, adjacency, uniform weights, cold start, ARAP energy with one global s,
solver, iteration cap, overlap detector, analysis, and the registered rule.

Columns per garment (identical code path): ARAP-0 (point, s=1) | S3-point | S3-ring.

Run: python -m tryon.checkpoint5_S3_point [--garment-ids ID [ID ...]]
Default ids: the golden garment and 831098a3-8778-480f-9a11-aa9df826d0f2. Console + one panel per garment.
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image, ImageDraw

from tryon.avatar_representation import build_avatar_representation
from tryon.garment_representation import build_garment_representation
from tryon.piecewise_continuous_test import solve_similarity_2point
from tryon.checkpoint5_ARAP0_deformation_field import (
    build_dense_mesh, build_adjacency, run_arap, local_step,
    GOLDEN_GARMENT_ID, AVATAR_USER_ID, MAX_ITERATIONS,
)
from tryon.checkpoint5_ARAP1 import (
    get_anatomical_handles, build_rings, analyze, hop_hist, multi_bfs, render_mesh, BIG_JUMP_DEG,
)
from tryon.checkpoint5_ARAP1_R2 import pair_by_x, robust_ratio, apply_T, draw_wire
from tryon.checkpoint5_S3 import (
    run_arap_scaled, arap_energy, directed_edges, far_ratios, classify_generalization, s_star,
)

OUTPUT_DIR = "outputs"
SECOND_GARMENT_ID = "831098a3-8778-480f-9a11-aa9df826d0f2"

# Values you recorded from earlier real runs; used only to flag silent drift on your machine.
KNOWN = {
    GOLDEN_GARMENT_ID: dict(ring_s=0.37750, ring_E=4964.3161, arap0_overlaps=39, arap0_inverted=5, arap0_rotmax=158.60),
    SECOND_GARMENT_ID: dict(ring_s=0.41072, ring_E=19524.0037),
}

OUT_CLEAN = "A — RINGS NOT NEEDED (S3-point is a clean pass)"
OUT_WEAK = "A-WEAK — contraction does not need rings, but rings still give a cleaner near field"
OUT_B = "B — RINGS STILL DOING WORK (S3-point does not pass the registered rule)"
OUT_UNDET = "UNDETERMINED (a ring baseline was unavailable)"

REGISTERED = """
======================================================================
REGISTERED BEFORE RUNNING (S3-point)
======================================================================
Question: once ARAP has a global scale DOF, are the R2 similarity rings still needed, or were they
scaffolding for the pre-scale problem? Only the constraint set changes (4 point handles, exact
avatar targets); no ring-derived soft constraints.

Per garment, S3-point is classified by the SAME registered rule used for S3-ring:
  PASS     outcome CONTRACTION (robust width, robust height, far-field mean all below THIS garment's MID)
           AND handle residual < 1e-6 AND S3-point overlap pairs / inverted triangles / max rotation jump
           are each <= ARAP-0's (the no-scale counterpart with the same four handles) AND energy never increases
  PARTIAL  far-field mean below MID but one or more PASS conditions fail
  FAIL     far-field mean NOT below MID
  (INVALID RUN if the handle residual is not ~0 or the energy increased)
Separately (comparison, not a pass condition): S3-point is AS CLEAN AS RINGS on a garment if it is <= S3-ring on
ALL FOUR of: overlap pairs, inverted triangles, max neighbor rotation jump, number of rotation jumps > 45 deg.

Registered three-way outcome across the tested garments:
  A       every garment PASSES and is AS CLEAN AS RINGS            -> rings were scaffolding; remove them
  A-WEAK  every garment PASSES but at least one is NOT as clean    -> rings are not needed for contraction,
                                                                      but they still buy a cleaner near field
  B       any garment is PARTIAL / FAIL / INVALID                  -> rings still do real work; a geometrically
                                                                      bounded handle region becomes the next question
No confident prediction is registered for A vs B. ARAP-0's point-handle pathologies were confounded with
missing scale, which is exactly why this cell is informative.
Informational (printed, not scored): diagnostic-point placement vs the T_underarm baseline (with point handles
nothing near the hem is pinned, so this is NOT an anchor effect), energy by mesh half, extreme edges.
"""


def final_outcome(results):
    """results: list of dicts with 'label' and 'as_clean' (True / False / None)."""
    if any(r["label"] != "PASS" for r in results):
        return OUT_B
    if any(r["as_clean"] is None for r in results):
        return OUT_UNDET
    if all(r["as_clean"] for r in results):
        return OUT_CLEAN
    return OUT_WEAK


def run_one(gid, avatar):
    bar = "=" * 70
    print(f"\n{bar}\nGARMENT {gid}\n{bar}")
    try:
        garment = build_garment_representation(gid, AVATAR_USER_ID)
    except Exception as exc:
        print(f"NOT TESTABLE: garment representation failed: {type(exc).__name__}: {exc}")
        return None
    if not garment.shoulders_available:
        print("NOT TESTABLE: shoulders_available=False (the four-handle design needs shoulders).")
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
    canvas_w, canvas_h = avatar.image.size
    shoulder_y = float(min(avatar.left_shoulder[1], avatar.right_shoulder[1]))
    hh = multi_bfs(adj, anat, n)

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
    print(f"Mesh: {n} vertices, {len(triangles)} triangles; handles {sorted(anat)}")
    print(f"T_underarm scale={scale_ua:.4f}  T_shoulder scale={scale_sh:.4f}  MID={MID:.4f}  required band=[{lo:.3f}, {hi:.3f}]")

    # ---- S3-ring comparison inputs (identical construction to the S3 / R2 scripts) ----
    rings, problems = build_rings(adj, handle_targets)
    have_rings = not problems
    ring_set, r2_targets = set(), {}
    if have_rings:
        T_of = {}
        for nm in ("left_underarm", "right_underarm", "left_shoulder", "right_shoulder"):
            T_of[landmark_index[nm]] = T_ua if "underarm" in nm else T_sh
        ring_set = set().union(*rings.values())
        for h, ring in rings.items():
            for v in ring:
                r2_targets[v] = handle_targets[h].copy() if v == h else apply_T(T_of[h], rest[v])
    else:
        print("NOTE: ring validation failed on this mesh; the S3-ring comparison column is unavailable: " + "; ".join(problems))

    # ---- solves ----
    print(f"\nRunning ARAP-0 (4 point handles, s fixed at 1, {MAX_ITERATIONS} it)...")
    def0, rot0 = run_arap(rest, adj, anat, handle_targets, MAX_ITERATIONS)
    print("Integrity check 1: S3 solver with s frozen at 1 and point handles vs the frozen ARAP-0 run...")
    chk, _, _, _, _ = run_arap_scaled(rest, adj, anat, handle_targets, MAX_ITERATIONS, freeze_s=True)
    max_diff = float(np.abs(chk - def0).max())
    print(f"  max |position difference| = {max_diff:.3e} px   ({'PASS' if max_diff < 1e-6 else 'FAIL'})")
    if max_diff >= 1e-6:
        print("  Stopping: S3-point would not be attributable to s alone.")
        sys.exit(1)
    print(f"Running S3-point (4 point handles, s free, {MAX_ITERATIONS} it)...")
    defp, rotp, s_p, trace, clamp_hits = run_arap_scaled(rest, adj, anat, handle_targets, MAX_ITERATIONS)
    defr = rotr = None
    s_r = None
    if have_rings:
        print("Running S3-ring (comparison)...")
        defr, rotr, s_r, trace_r, _ = run_arap_scaled(rest, adj, ring_set, r2_targets, MAX_ITERATIONS)

    I, J = directed_edges(adj)
    E0 = arap_energy(rest, def0, rot0, 1.0, I, J)
    Ep = arap_energy(rest, defp, rotp, s_p, I, J)
    Er = arap_energy(rest, defr, rotr, s_r, I, J) if have_rings else None

    # ---- drift guards against your recorded real values ----
    print(f"\n{'-'*70}\nREPRODUCTION OF RECORDED RESULTS (flags silent drift)\n{'-'*70}")
    print(f"  this run: ARAP-0 E(s=1)={E0:.4f} | S3-point s={s_p:.5f} E={Ep:.4f}" + (f" | S3-ring s={s_r:.5f} E={Er:.4f}" if have_rings else ""))
    kn = KNOWN.get(gid)
    if kn is None:
        print("  no recorded values for this garment id")
    else:
        if have_rings and "ring_s" in kn:
            ok = abs(s_r - kn["ring_s"]) < 6e-6 and abs(Er - kn["ring_E"]) < 2e-6 * kn["ring_E"] + 1e-3
            print(f"  S3-ring: s={s_r:.5f} E={Er:.4f}  vs recorded s={kn['ring_s']:.5f} E={kn['ring_E']:.4f}  -> {'MATCH' if ok else 'MISMATCH — the ring baseline differs from the recorded S3 result'}")
        if "arap0_overlaps" in kn:
            tmpA = analyze(rest, def0, rot0, triangles, adj, anat, anat, handle_targets, canvas_w, shoulder_y)
            dmax = max(e[2] for e in tmpA["rot_edges"])
            ok = (len(tmpA["overlaps"]) == kn["arap0_overlaps"] and len(tmpA["inverted"]) == kn["arap0_inverted"]
                  and abs(dmax - kn["arap0_rotmax"]) < 0.05)
            print(f"  ARAP-0: overlaps={len(tmpA['overlaps'])} inverted={len(tmpA['inverted'])} max rot jump={dmax:.2f}  vs recorded "
                  f"{kn['arap0_overlaps']} / {kn['arap0_inverted']} / {kn['arap0_rotmax']:.2f}  -> {'MATCH' if ok else 'MISMATCH'}")

    # ---- S3-point convergence ----
    print(f"\n{'-'*70}\nS3-POINT CONVERGENCE (iterations 1-5, then every 5th; full run is {MAX_ITERATIONS} it)\n{'-'*70}")
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
    print(f"  Integrity check 2: iterations where E INCREASED = {increases}   ({'PASS' if increases == 0 else 'FAIL'})")
    print(f"  s clamp activations = {clamp_hits}")
    R_fin = local_step(rest, defp, adj, n)
    s_chk = s_star(rest, defp, R_fin, I, J)
    print(f"  Integrity check 3: final s = {s_p:.5f}; s-step re-solved on the final configuration = {s_chk:.5f}  (relative gap {abs(s_chk - s_p) / s_p:.2e})")
    print(f"  first iteration with dE_rel < 1e-3 / 1e-4 / 1e-5: {first_below[1e-3]} / {first_below[1e-4]} / {first_below[1e-5]}")
    band = "BELOW the required band" if s_p < lo else ("INSIDE the required band" if s_p <= hi else "ABOVE the required band")
    print(f"  final s = {s_p:.4f} -> {band} [{lo:.3f}, {hi:.3f}]   (S3-ring s = {('%.4f' % s_r) if have_rings else 'n/a'}; ARAP-0 s fixed at 1)")

    # ---- analysis, identical code path ----
    A0 = analyze(rest, def0, rot0, triangles, adj, anat, anat, handle_targets, canvas_w, shoulder_y)
    Ap = analyze(rest, defp, rotp, triangles, adj, anat, anat, handle_targets, canvas_w, shoulder_y)
    Ar = analyze(rest, defr, rotr, triangles, adj, ring_set, anat, r2_targets, canvas_w, shoulder_y) if have_rings else None
    cols = [("ARAP-0", def0, A0, 1.0), ("S3-point", defp, Ap, s_p)]
    if have_rings:
        cols.append(("S3-ring", defr, Ar, s_r))
    names = [c[0] for c in cols]
    RB = {k: (robust_ratio(rest, d, 0), robust_ratio(rest, d, 1)) for k, d, a, s in cols}
    FR = {k: far_ratios(rest, d, adj, hh) for k, d, a, s in cols}
    ROT = {k: np.array([e[2] for e in a["rot_edges"]]) for k, d, a, s in cols}

    def row(label, f, w=15):
        print(f"  {label:<48}" + "".join(f"{f(k):>{w}}" for k in names))

    print(f"\n{'='*70}\nEIGHT CHECKS (identical on every garment)  garment={gid}\n{'='*70}")
    print(f"  {'':<48}" + "".join(f"{k:>15}" for k in names))
    row("1 far-field mean edge ratio (3+ hops)", lambda k: f"{FR[k].mean():.3f}")
    row("1 far-field median edge ratio", lambda k: f"{np.median(FR[k]):.3f}")
    row("2 robust width ratio", lambda k: f"{RB[k][0]:.3f}")
    row("2 robust height ratio", lambda k: f"{RB[k][1]:.3f}")
    row("3 max constraint residual (px)", lambda k: f"{dict((c[0], c[2]) for c in cols)[k]['residual']:.6f}")
    row("4 overlap pairs", lambda k: f"{len(dict((c[0], c[2]) for c in cols)[k]['overlaps'])}")
    row("4 max / total overlap area (px^2)", lambda k: (lambda a: f"{max([o[2] for o in a['overlaps']], default=0):.1f}/{sum(o[2] for o in a['overlaps']):.1f}")(dict((c[0], c[2]) for c in cols)[k]))
    row("5 inverted triangles", lambda k: f"{len(dict((c[0], c[2]) for c in cols)[k]['inverted'])}")
    row("6 max neighbor rotation jump (deg)", lambda k: f"{ROT[k].max():.1f}")
    row("6 rotation jumps > 45 deg", lambda k: f"{int((ROT[k] > BIG_JUMP_DEG).sum())}")
    row("7 garment top above avatar shoulder line (px)", lambda k: f"{dict((c[0], c[2]) for c in cols)[k]['bbox']['above_shoulder']:.1f}")
    row("8 global scale s", lambda k: ("1 (fixed)" if k == "ARAP-0" else f"{dict((c[0], c[3]) for c in cols)[k]:.4f}"))
    print(f"  (pair scales: underarm {scale_ua:.4f}, shoulder {scale_sh:.4f}; MID {MID:.4f})")

    print(f"\n  EDGE-LENGTH RATIO percentiles, all edges (0/5/25/50/75/95/100):")
    for k, d, a, s in cols:
        print(f"    {k:<9}" + "  ".join(f"{v:6.3f}" for v in np.percentile(a["ratios"], [0, 5, 25, 50, 75, 95, 100])) + f"   mean={a['ratios'].mean():.3f}")
    print(f"  FAR-FIELD (>=3 hops) edge-ratio P5/P25/P50/P75/P95:")
    for k, d, a, s in cols:
        print(f"    {k:<9}" + " / ".join(f"{v:.3f}" for v in np.percentile(FR[k], [5, 25, 50, 75, 95])) + f"   (median/s = {np.median(FR[k]) / s:.3f})")

    print(f"\n  Extreme edges for S3-point (H=handle, O=other; hh = hops to nearest handle):")
    edges = []
    for i in adj:
        for j in adj[i]:
            if j > i:
                rl = float(np.linalg.norm(rest[i] - rest[j]))
                if rl > 1e-9:
                    edges.append((float(np.linalg.norm(defp[i] - defp[j])) / rl, i, j, rl))
    edges.sort()
    lab = lambda v: "H" if v in anat else "O"
    for tag, rows_ in (("5 most COMPRESSED", edges[:5]), ("5 most STRETCHED", edges[-5:][::-1])):
        print(f"    {tag}:")
        for r, i, j, rl in rows_:
            print(f"      ratio={r:6.3f} edge({i:3d},{j:3d}) {lab(i)}-{lab(j)} hh=({hh[i]},{hh[j]}) rest=({rest[i][0]:.0f},{rest[i][1]:.0f})-({rest[j][0]:.0f},{rest[j][1]:.0f}) len={rl:.1f}px")

    # ---- energy by mesh half (constraint-agnostic) ----
    xs_h = np.array([rest[h][0] for h in anat])
    split_x = float(xs_h.mean())
    def half_energy(cur, R, s):
        tot = {"small-x": 0.0, "large-x": 0.0}
        for i in adj:
            for j in adj[i]:
                d = (cur[i] - cur[j]) - s * (R[i] @ (rest[i] - rest[j]))
                mid_x = (rest[i][0] + rest[j][0]) / 2
                tot["small-x" if mid_x < split_x else "large-x"] += float(d @ d)
        return tot
    print(f"\n  ENERGY BY MESH HALF (directed-edge terms split by rest midpoint x at the mean handle x = {split_x:.0f}; same definition for every column):")
    for k, d, a, s in cols:
        R = {"ARAP-0": rot0, "S3-point": rotp, "S3-ring": rotr}[k]
        h = half_energy(d, R, s)
        print(f"    {k:<9} small-x={h['small-x']:11.1f}  large-x={h['large-x']:11.1f}  ratio={h['small-x'] / h['large-x']:.2f}x   total={h['small-x'] + h['large-x']:.1f}")

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
        parts = []
        for k, d, a, s in cols[1:]:
            parts.append(f"({d[v][0]:.0f},{d[v][1]:.0f}) {np.linalg.norm(d[v] - base):.0f}px")
        pin = " [PINNED in S3-ring]" if v in ring_set else ""
        print(f"    {nm:19s} v{v:3d}: ({rest[v][0]:.0f},{rest[v][1]:.0f}) | ({base[0]:.0f},{base[1]:.0f}) | " + " | ".join(parts) + pin)

    print(f"\n  WHERE THE PATHOLOGY LIVES (hops from handles 0, 1, 2, 3+):")
    for k, d, a, s in cols:
        big = [e for e in a["rot_edges"] if e[2] > BIG_JUMP_DEG]
        print(f"    {k:<9} overlap pairs hh={hop_hist([o[4] for o in a['overlaps']])}  inverted tris hh={hop_hist(a['inv_hops_h'])}  rot jumps>{BIG_JUMP_DEG:.0f} hh={hop_hist([e[4] for e in big])}")

    # ---- registered classification ----
    wp, hp_ = RB["S3-point"]
    fp = FR["S3-point"].mean()
    below = [nm for nm, v in (("width", wp), ("height", hp_), ("far-field", fp)) if v < MID]
    label, failing = classify_generalization(
        contraction=(len(below) == 3), far_ok=(fp < MID),
        res_ok=(Ap["residual"] < 1e-6),
        ov_ok=(len(Ap["overlaps"]) <= len(A0["overlaps"])),
        inv_ok=(len(Ap["inverted"]) <= len(A0["inverted"])),
        rot_ok=(float(ROT["S3-point"].max()) <= float(ROT["ARAP-0"].max())),
        integrity_ok=(increases == 0))
    print(f"\n{'='*70}\nREGISTERED CLASSIFICATION (S3-point vs its no-scale counterpart ARAP-0)  MID={MID:.4f}\n{'='*70}")
    print(f"  S3-point: robust-width={wp:.3f} robust-height={hp_:.3f} far-field={fp:.3f}  (below MID: {', '.join(below) if below else 'none'})")
    print(f"  garment {gid} -> {label}")
    for f in failing:
        print(f"    - {f}")
    as_clean = None
    if have_rings:
        items = [
            ("overlap pairs", len(Ap["overlaps"]), len(Ar["overlaps"])),
            ("inverted triangles", len(Ap["inverted"]), len(Ar["inverted"])),
            ("max rotation jump", float(ROT["S3-point"].max()), float(ROT["S3-ring"].max())),
            (f"rotation jumps > {BIG_JUMP_DEG:.0f}", int((ROT["S3-point"] > BIG_JUMP_DEG).sum()), int((ROT["S3-ring"] > BIG_JUMP_DEG).sum())),
        ]
        as_clean = all(a <= b for _, a, b in items)
        print(f"  AS CLEAN AS RINGS (S3-point <= S3-ring on all four)? {'YES' if as_clean else 'NO'}")
        for nm, a, b in items:
            print(f"    {nm:<26} S3-point {a:>8.1f}   S3-ring {b:>8.1f}   {'ok' if a <= b else 'WORSE'}")

    # ---- panel ----
    print("\nRendering panel (S3-point render | S3-point wireframe | S3-ring wireframe)...")
    garment_arr = np.array(garment.image.convert("RGBA"))
    avatar_rgba = np.array(avatar.image.convert("RGBA"))
    layer = render_mesh(rest, defp, triangles, garment_arr, canvas_w, canvas_h)
    comp = avatar_rgba.copy()
    alpha = layer[:, :, 3:4].astype(float) / 255.0
    comp[:, :, :3] = (avatar_rgba[:, :, :3] * (1 - alpha) + layer[:, :, :3] * alpha).astype(np.uint8)
    left = Image.fromarray(comp).convert("RGB")
    dr = ImageDraw.Draw(left)
    for t in handle_targets.values():
        dr.ellipse([t[0] - 5, t[1] - 5, t[0] + 5, t[1] + 5], outline=(255, 255, 0), width=2)
    dr.text((6, 6), f"S3-point render (s={s_p:.3f})", fill=(255, 255, 255))
    base_poly = [tuple(apply_T(T_ua, rest[i])) for i in range(n_boundary)]
    mid_img = draw_wire(defp, triangles, set(), anat, canvas_w, canvas_h, shoulder_y, "S3-point wireframe")
    ImageDraw.Draw(mid_img).polygon(base_poly, outline=(255, 0, 255))
    panel = Image.new("RGB", (canvas_w * 3, canvas_h), (0, 0, 0))
    panel.paste(left, (0, 0))
    panel.paste(mid_img, (canvas_w, 0))
    if have_rings:
        right_img = draw_wire(defr, triangles, ring_set, anat, canvas_w, canvas_h, shoulder_y, "S3-ring wireframe")
        ImageDraw.Draw(right_img).polygon(base_poly, outline=(255, 0, 255))
        panel.paste(right_img, (canvas_w * 2, 0))
    out_path = f"{OUTPUT_DIR}/checkpoint5_S3point_panel_{gid[:8]}.png"
    panel.save(out_path)
    print(f"Saved: {out_path}")
    print("  red = canvas, green = mesh, orange = ring vertices (S3-ring panel only), yellow = handles, cyan = shoulder line, magenta = T_underarm baseline outline")

    return dict(gid=gid, label=label, as_clean=as_clean, s_point=s_p, s_ring=s_r, far_point=float(fp),
                far_ring=(float(FR["S3-ring"].mean()) if have_rings else None), failing=failing)


def main(ids):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(__doc__)
    print(REGISTERED)
    print("Building the avatar once...")
    try:
        avatar = build_avatar_representation()
    except Exception as exc:
        print(f"avatar build failed: {type(exc).__name__}: {exc}")
        sys.exit(2)
    results = [r for r in (run_one(g, avatar) for g in ids) if r is not None]
    if not results:
        print("\nNo garment was testable.")
        sys.exit(2)
    bar = "=" * 70
    print(f"\n{bar}\nREGISTERED THREE-WAY OUTCOME (across {len(results)} tested garment(s))\n{bar}")
    for r in results:
        print(f"  {r['gid'][:8]}  S3-point: {r['label']:<8} as clean as rings: {str(r['as_clean']):<5} "
              f"s(point)={r['s_point']:.4f} s(ring)={('%.4f' % r['s_ring']) if r['s_ring'] is not None else 'n/a'}  "
              f"far-field(point)={r['far_point']:.3f}" + (f"  failing: {r['failing']}" if r["failing"] else ""))
    print(f"\n  OUTCOME: {final_outcome(results)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="S3 with point handles on one or more garments.")
    ap.add_argument("--garment-ids", nargs="*", default=[GOLDEN_GARMENT_ID, SECOND_GARMENT_ID])
    args = ap.parse_args()
    main(args.garment_ids)