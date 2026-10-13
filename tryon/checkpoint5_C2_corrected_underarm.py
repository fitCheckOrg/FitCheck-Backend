"""
tryon/checkpoint5_C2_corrected_underarm.py

EXPERIMENT C2 — CORRECTED-UNDERARM BASELINE. Exactly ONE input changes:

    avatar underarm landmarks:  x = unchanged,  y = the independently measured silhouette armpit apex y
    (apex from checkpoint5_landmark_audit.avatar_audit: the row where each arm first separates from the torso)

The change is applied through a proxy object; the avatar module is not touched. If the apex is not found on both
sides the script STOPS (no silent fallback). Everything else is imported and unchanged: avatar, garments, garment
underarms, the similarity solver (solve_similarity_2point), TopMesh construction (incl. the shoulderless shim),
renderer, and the 4-handle S3-point experiment.

Per garment the script shows, side by side, the ORIGINAL and the CORRECTED avatar underarms:
    similarity placement (closed form, through the garment's two underarm landmarks)
    4-handle S3-point (only for garments with shoulders; shoulder targets are unchanged)

REGISTERED BEFORE RUNNING
  Arithmetic note: once the avatar underarm line is level, the similarity rotation is simply minus the garment's own
  underarm-line angle. The rotation numbers therefore confirm the plumbing; they are not a test of the hypothesis.
  C2-a  corrected avatar underarm line angle is 0.00 deg (the two apex y values are equal).
  C2-b  corrected similarity rotation = -(garment underarm-line angle): +0.23 deg golden, +0.47 deg second, -2.39 deg
        white tee (tolerance 0.01 deg). Original rotations must reproduce the recorded -8.77 / -8.53 / -11.40.
  P1    TILT REMOVAL: tilt error = (mean local rotation of the placed garment) - (axis-implied rotation
        phi_garment - phi_avatar, from the landmark audit). For the similarity the mean local rotation IS theta;
        for the 4-handle runs it is the circular mean of the solver's per-vertex rotations. P1 (similarity): the
        error after correction is below 0.5x its original value AND <= |garment underarm-line angle| + 1.5 deg
        (the garment's own tilt plus slack; thresholds mine).
        REVISION (before any real data): P1 was first drafted on the rendered silhouette's centreline lean. A sandbox
        test showed that measurement is biased for tilted placements (the tilted hem truncates the row chords: it
        recovered a 4.9 deg reduction where the analytic value was 9.0 deg), so the analytic definition replaces it.
  P2    the garment's underarm landmarks, after the similarity, lie within 3 px of the independent silhouette apexes
        (original: 6.3 px / 10.0 px).
  Informational (no pass/fail; the look is judged by eye on the panels):
    mean local rotation and tilt error for every column; top-above-shoulder-line; hem row and hem minus hip line;
    TORSO COVERAGE  = garment pixels on the avatar torso mask / torso mask pixels, rows shoulder line..hem
    SPILL           = garment pixels outside the avatar's whole-body silhouette / garment pixels
    UPPER-ARM COVERAGE = garment pixels on the upper-arm mask / upper-arm mask pixels, rows shoulder line..elbow line
                         (a proxy for sleeve appearance: collapsed sleeves cover less of the arms)
    SHOULDER-BAND EXPOSURE = uncovered fraction of (torso or upper-arm) mask in rows shoulder line-12..+20
  Drift guards: original rotations and the original 4-handle S3-point (s, E) against your recorded values.

Run: python -m tryon.checkpoint5_C2_corrected_underarm [--garment-ids ID [ID ...]]
Default ids: golden, 831098a3-..., white tee 7e16734e-...   Output: console + one panel per garment.
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
    build_dense_mesh, build_adjacency, trace_and_simplify, GOLDEN_GARMENT_ID, AVATAR_USER_ID, MAX_ITERATIONS,
)
from tryon.checkpoint5_ARAP1 import get_anatomical_handles, render_mesh
from tryon.checkpoint5_ARAP1_R2 import pair_by_x, apply_T
from tryon.checkpoint5_S3 import run_arap_scaled, arap_energy, directed_edges
from tryon.checkpoint5_S3_underarm import _ShoulderlessShim, tolerant_shoulders, WHITE_TEE_ID, SECOND_GARMENT_ID
from tryon.checkpoint5_landmark_audit import avatar_audit, garment_audit, ang

OUTPUT_DIR = "outputs"
KNOWN_ROT = {GOLDEN_GARMENT_ID: -8.77, SECOND_GARMENT_ID: -8.53, WHITE_TEE_ID: -11.40}
KNOWN_4H = {GOLDEN_GARMENT_ID: (0.36869, 1860.5798), SECOND_GARMENT_ID: (0.36923, 3986.4289)}


class CorrectedAvatar:
    """Proxy: every attribute comes from the real avatar except the two underarm landmarks."""
    def __init__(self, base, left_underarm, right_underarm):
        self._base = base
        self.left_underarm = left_underarm
        self.right_underarm = right_underarm

    def __getattr__(self, name):
        return getattr(self._base, name)


def corrected_underarms(avatar, A):
    apex = A["apex"]
    if not (apex["small-x"] and apex["large-x"]):
        return None
    lu, ru = avatar.left_underarm, avatar.right_underarm
    if lu[0] < ru[0]:
        return (lu[0], apex["small-x"][1]), (ru[0], apex["large-x"][1])
    return (lu[0], apex["large-x"][1]), (ru[0], apex["small-x"][1])


def check_p1(err_before, err_after, garment_line_deg):
    return bool(abs(err_after) < 0.5 * abs(err_before) and abs(err_after) <= abs(garment_line_deg) + 1.5)


def check_p2(dist_after):
    return bool(max(dist_after) <= 3.0)


def fit_metrics(layer, masks, avatar, shoulder_y):
    Gm = layer[:, :, 3] >= 128
    H, W = Gm.shape
    ys = np.where(Gm.any(axis=1))[0]
    out = {}
    if ys.size == 0:
        return None
    y_top, y_hem = int(ys.min()), int(ys.max())
    ys0 = int(shoulder_y)
    Tm, Um, Bm = masks["torso"], masks["arms"], masks["body"]
    rows = slice(ys0, y_hem + 1)
    out["y_top"], out["y_hem"] = y_top, y_hem
    out["above_shoulder"] = float(shoulder_y - y_top)
    hips_y = float(np.mean([avatar.left_hip[1], avatar.right_hip[1]]))
    out["hem_minus_hips"] = float(y_hem - hips_y)
    t_den = Tm[rows].sum()
    out["torso_cov"] = float((Gm & Tm)[rows].sum() / t_den) if t_den else float("nan")
    out["spill"] = float((Gm & ~Bm).sum() / max(Gm.sum(), 1))
    elbow_y = int(np.mean([avatar.left_elbow[1], avatar.right_elbow[1]]))
    arow = slice(ys0, elbow_y + 1)
    a_den = Um[arow].sum()
    out["arm_cov"] = float((Gm & Um)[arow].sum() / a_den) if a_den else float("nan")
    band = slice(int(shoulder_y) - 12, int(shoulder_y) + 21)
    b_den = (Tm | Um)[band].sum()
    out["shoulder_exposed"] = float(1.0 - (Gm & (Tm | Um))[band].sum() / b_den) if b_den else float("nan")
    return out


def run_garment(gid, avatar, avatar_c, A, masks, canvas_w, canvas_h, shoulder_y):
    bar = "=" * 70
    print(f"\n{bar}\nGARMENT {gid}\n{bar}")
    try:
        with tolerant_shoulders():
            garment = garment_module.build_garment_representation(gid, AVATAR_USER_ID)
    except Exception as exc:
        print(f"NOT TESTABLE: garment representation failed: {type(exc).__name__}: {exc}")
        return None
    has_sh = bool(garment.shoulders_available)
    try:
        if has_sh:
            rest, triangles, li, nb = build_dense_mesh(garment)
        else:
            b0 = trace_and_simplify(garment.mask, epsilon=2.0)
            rest, triangles, li, nb = build_dense_mesh(_ShoulderlessShim(garment, tuple(b0[0])))
    except Exception as exc:
        print(f"NOT TESTABLE: mesh construction failed: {type(exc).__name__}: {exc}")
        return None
    n = len(rest)
    adj = build_adjacency(n, triangles)
    g_ua = pair_by_x(garment.left_underarm, garment.right_underarm)
    GA = garment_audit(garment)
    ang_g = GA["ang_underarm"]
    phi_g = GA["axis"][0] if GA["axis"] else None
    phi_a = A["axis"][0] if A["axis"] else None
    theta_axis = (phi_g - phi_a) if (phi_g is not None and phi_a is not None) else None
    print(f"  garment size={garment.image.size}; shoulders_available={has_sh}; mesh {n} vertices / {len(triangles)} triangles; "
          f"garment underarm-line angle {ang_g:+.2f} deg")

    variants = []      # (key, label, deformed, targets-for-drawing, theta, scale)
    ua_o = pair_by_x(avatar.left_underarm, avatar.right_underarm)
    ua_c = pair_by_x(avatar_c.left_underarm, avatar_c.right_underarm)
    sims = {}
    for key, ua in (("orig", ua_o), ("corr", ua_c)):
        T = np.array(solve_similarity_2point(list(g_ua), list(ua)))
        sims[key] = dict(T=T, theta=float(np.degrees(np.arctan2(T[1, 0], T[0, 0]))), scale=float(np.hypot(T[0, 0], T[1, 0])),
                         deformed=np.array([apply_T(T, p) for p in rest]), ua=ua)
        sims[key]["ua_img"] = [apply_T(T, p) for p in g_ua]
    drift_ok = abs(sims["orig"]["theta"] - KNOWN_ROT.get(gid, sims["orig"]["theta"])) < 0.01
    print(f"  original similarity rotation {sims['orig']['theta']:+.2f} deg"
          + (f"  vs recorded {KNOWN_ROT[gid]:+.2f}  -> {'MATCH' if drift_ok else 'MISMATCH'}" if gid in KNOWN_ROT else ""))
    exp_theta = -ang_g
    c2b = abs(sims["corr"]["theta"] - exp_theta) < 0.01
    print(f"  C2-b corrected similarity rotation {sims['corr']['theta']:+.2f} deg; expected -(garment angle) = {exp_theta:+.2f}  -> {'PASS' if c2b else 'FAIL'}")

    four = {}
    if has_sh:
        I, J = directed_edges(adj)
        for key, av in (("orig", avatar), ("corr", avatar_c)):
            t4 = get_anatomical_handles(garment, av, li)
            d4, r4, s4, tr, _ = run_arap_scaled(rest, adj, set(t4), t4, MAX_ITERATIONS)
            angs = np.degrees(np.arctan2(r4[:, 1, 0], r4[:, 0, 0]))
            mean_rot = float(np.degrees(np.arctan2(np.mean(np.sin(np.radians(angs))), np.mean(np.cos(np.radians(angs))))))
            four[key] = dict(deformed=d4, targets=t4, s=s4, E=arap_energy(rest, d4, r4, s4, I, J), mean_rot=mean_rot, rot_std=float(np.std(angs)))
        kn = KNOWN_4H.get(gid)
        if kn:
            ok = abs(four["orig"]["s"] - kn[0]) < 6e-6 and abs(four["orig"]["E"] - kn[1]) < 2e-6 * kn[1] + 1e-3
            print(f"  original 4-handle S3-point: s={four['orig']['s']:.5f} E={four['orig']['E']:.4f}  vs recorded {kn[0]:.5f} / {kn[1]:.4f}  -> {'MATCH' if ok else 'MISMATCH'}")
        print(f"  corrected 4-handle S3-point: s={four['corr']['s']:.5f} E={four['corr']['E']:.4f}")

    garment_arr = np.array(garment.image.convert("RGBA"))
    avatar_rgba = np.array(avatar.image.convert("RGBA"))
    cols = [("sim-orig", "similarity, ORIGINAL underarms", sims["orig"]["deformed"],
             {0: ua_o[0], 1: ua_o[1]}, sims["orig"]["theta"]),
            ("sim-corr", "similarity, CORRECTED underarms", sims["corr"]["deformed"],
             {0: ua_c[0], 1: ua_c[1]}, sims["corr"]["theta"])]
    if has_sh:
        cols += [("4h-orig", "4-handle, ORIGINAL underarms", four["orig"]["deformed"], four["orig"]["targets"], None),
                 ("4h-corr", "4-handle, CORRECTED underarms", four["corr"]["deformed"], four["corr"]["targets"], None)]
    M = {}
    renders = {}
    for key, label, defm, tg, theta in cols:
        layer = render_mesh(rest, defm, triangles, garment_arr, canvas_w, canvas_h)
        M[key] = fit_metrics(layer, masks, avatar, shoulder_y)
        which = "corr" if "corr" in key else "orig"
        mrot = sims[which]["theta"] if key.startswith("sim") else four[which]["mean_rot"]
        M[key]["mean_rot"] = mrot
        M[key]["tilt_err"] = (mrot - theta_axis) if theta_axis is not None else None
        comp = avatar_rgba.copy()
        alpha = layer[:, :, 3:4].astype(float) / 255.0
        comp[:, :, :3] = (avatar_rgba[:, :, :3] * (1 - alpha) + layer[:, :, :3] * alpha).astype(np.uint8)
        im = Image.fromarray(comp).convert("RGB")
        d = ImageDraw.Draw(im)
        xc = float(np.mean([avatar.left_shoulder[0], avatar.right_shoulder[0]]))
        d.line([(xc, 0), (xc, canvas_h)], fill=(0, 140, 190), width=1)
        d.line([(0, shoulder_y), (canvas_w, shoulder_y)], fill=(0, 140, 190), width=1)
        if isinstance(tg, dict) and all(isinstance(v, np.ndarray) or isinstance(v, tuple) for v in tg.values()):
            for t in tg.values():
                d.ellipse([t[0] - 5, t[1] - 5, t[0] + 5, t[1] + 5], outline=(255, 255, 0), width=2)
        d.text((4, 4), label, fill=(255, 255, 255))
        if M[key]["tilt_err"] is not None:
            d.text((4, 16), f"mean rotation {M[key]['mean_rot']:+.1f}  tilt err {M[key]['tilt_err']:+.1f} deg", fill=(255, 255, 255))
        renders[key] = im

    names = [c[0] for c in cols]
    def row(label, f, w=13):
        print(f"  {label:<46}" + "".join(f"{f(k):>{w}}" for k in names))
    print(f"\n  {'':<46}" + "".join(f"{k:>13}" for k in names))
    row("avatar underarm line angle used (deg)", lambda k: f"{ang(sims['corr' if 'corr' in k else 'orig']['ua'][0], sims['corr' if 'corr' in k else 'orig']['ua'][1]):+.2f}")
    row("similarity rotation (deg)", lambda k: (f"{sims['corr' if 'corr' in k else 'orig']['theta']:+.2f}" if k.startswith("sim") else "-"))
    row("similarity scale", lambda k: (f"{sims['corr' if 'corr' in k else 'orig']['scale']:.4f}" if k.startswith("sim") else "-"))
    row("4-handle global scale s", lambda k: (f"{four['corr' if 'corr' in k else 'orig']['s']:.4f}" if k.startswith("4h") else "-"))
    row("mean local rotation (deg)", lambda k: f"{M[k]['mean_rot']:+.2f}")
    row("  tilt error vs axis-implied rotation (deg)", lambda k: (f"{M[k]['tilt_err']:+.2f}" if M[k]["tilt_err"] is not None else "n/a"))
    if any(k.startswith("4h") for k in names):
        print(f"  (4-handle local-rotation spread, std deg: " + ", ".join(f"{k}={four['corr' if 'corr' in k else 'orig']['rot_std']:.1f}" for k in names if k.startswith('4h')) + ")")
    row("top above avatar shoulder line (px)", lambda k: f"{M[k]['above_shoulder']:.1f}")
    row("hem row (y) / hem minus hip line (px)", lambda k: f"{M[k]['y_hem']}/{M[k]['hem_minus_hips']:+.0f}")
    row("torso coverage (shoulder line..hem)", lambda k: f"{M[k]['torso_cov']:.1%}")
    row("spill outside the body silhouette", lambda k: f"{M[k]['spill']:.1%}")
    row("upper-arm coverage (sleeve proxy)", lambda k: f"{M[k]['arm_cov']:.1%}")
    row("shoulder-band exposure", lambda k: f"{M[k]['shoulder_exposed']:.1%}")

    apex = A["apex"]
    apex_pair = [apex["small-x"], apex["large-x"]]
    d_o = [float(np.hypot(sims["orig"]["ua_img"][k][0] - apex_pair[k][0], sims["orig"]["ua_img"][k][1] - apex_pair[k][1])) for k in (0, 1)]
    d_c = [float(np.hypot(sims["corr"]["ua_img"][k][0] - apex_pair[k][0], sims["corr"]["ua_img"][k][1] - apex_pair[k][1])) for k in (0, 1)]
    print(f"\n  garment underarm landmarks after the similarity vs the independent silhouette apexes (small-x / large-x):"
          f" original {d_o[0]:.1f}px / {d_o[1]:.1f}px;  corrected {d_c[0]:.1f}px / {d_c[1]:.1f}px")
    p2 = check_p2(d_c)
    e_o, e_c = M["sim-orig"]["tilt_err"], M["sim-corr"]["tilt_err"]
    p1 = (check_p1(e_o, e_c, ang_g) if (e_o is not None and e_c is not None) else None)
    print(f"  axis-implied rotation (phi_garment - phi_avatar) = " + (f"{theta_axis:+.2f} deg" if theta_axis is not None else "n/a"))
    print(f"  P1 tilt removal (similarity): tilt error {e_o:+.2f} -> {e_c:+.2f} deg; bound |{ang_g:+.2f}|+1.5 = {abs(ang_g) + 1.5:.2f}  -> {'PASS' if p1 else ('FAIL' if p1 is not None else 'n/a')}")
    print(f"  P2 underarm vs apex <= 3 px after correction -> {'PASS' if p2 else 'FAIL'}")

    # panel
    panel = Image.new("RGB", (canvas_w * len(cols), canvas_h), (0, 0, 0))
    for i, (key, *_r) in enumerate(cols):
        panel.paste(renders[key], (canvas_w * i, 0))
    path = f"{OUTPUT_DIR}/checkpoint5_C2_panel_{gid[:8]}.png"
    panel.save(path)
    print(f"  Saved: {path}   (columns: " + " | ".join(c[1] for c in cols) + "; thin cyan guides = avatar centre line and shoulder line; yellow = targets)")
    return dict(gid=gid, has_sh=has_sh, theta_o=sims["orig"]["theta"], theta_c=sims["corr"]["theta"], c2b=c2b, drift_ok=drift_ok,
                err_o=e_o, err_c=e_c, p1=p1, p2=p2, d_o=d_o, d_c=d_c, M=M)


def main(ids):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(__doc__)
    print("Building the avatar...")
    avatar = build_avatar_representation()
    A = avatar_audit(avatar)
    cu = corrected_underarms(avatar, A)
    if cu is None:
        print("STOP: the silhouette armpit apex was not found on both sides; no correction is applied and nothing is run.")
        sys.exit(2)
    avatar_c = CorrectedAvatar(avatar, cu[0], cu[1])
    bar = "=" * 70
    print(f"\n{bar}\nTHE ONE INPUT CHANGE\n{bar}")
    for nm, o, c in (("left_underarm ", avatar.left_underarm, avatar_c.left_underarm), ("right_underarm", avatar.right_underarm, avatar_c.right_underarm)):
        print(f"  {nm}: ({o[0]:.1f}, {o[1]:.1f}) -> ({c[0]:.1f}, {c[1]:.1f})   (x kept, y = silhouette apex; dy = {c[1] - o[1]:+.1f}px)")
    ua_c = pair_by_x(avatar_c.left_underarm, avatar_c.right_underarm)
    ang_c = ang(ua_c[0], ua_c[1])
    print(f"  C2-a corrected avatar underarm line angle = {ang_c:+.2f} deg -> {'PASS' if abs(ang_c) < 0.01 else 'FAIL'}   (original {A['ang_underarm']:+.2f} deg)")
    ua_o = pair_by_x(avatar.left_underarm, avatar.right_underarm)
    print(f"  underarm span {np.hypot(ua_o[1][0] - ua_o[0][0], ua_o[1][1] - ua_o[0][1]):.1f}px -> {np.hypot(ua_c[1][0] - ua_c[0][0], ua_c[1][1] - ua_c[0][1]):.1f}px (scale is essentially unchanged)")

    canvas_w, canvas_h = avatar.image.size
    shoulder_y = float(min(avatar.left_shoulder[1], avatar.right_shoulder[1]))
    masks = {"torso": np.asarray(avatar.torso_mask) > 0, "arms": np.asarray(avatar.upper_arms_mask) > 0, "body": A["F"]}
    print(f"  avatar masks: torso {int(masks['torso'].sum())} px, upper arms {int(masks['arms'].sum())} px, body silhouette {int(masks['body'].sum())} px; "
          f"avatar torso axis lean {A['axis'][0] if A['axis'] else float('nan'):+.2f} deg")

    results = [r for r in (run_garment(g, avatar, avatar_c, A, masks, canvas_w, canvas_h, shoulder_y) for g in ids) if r is not None]
    if results:
        print(f"\n{bar}\nSUMMARY (similarity placement; informational fit numbers are per-garment above)\n{bar}")
        print(f"  {'garment':<12}{'rot orig':>10}{'rot corr':>10}{'tilt err orig':>15}{'tilt err corr':>15}{'P1':>6}{'P2':>6}{'C2-b':>6}{'drift':>7}")
        for r in results:
            print(f"  {r['gid'][:8]:<12}{r['theta_o']:>+10.2f}{r['theta_c']:>+10.2f}{r['err_o']:>+15.2f}{r['err_c']:>+15.2f}"
                  f"{('PASS' if r['p1'] else 'FAIL') if r['p1'] is not None else 'n/a':>6}{'PASS' if r['p2'] else 'FAIL':>6}{'PASS' if r['c2b'] else 'FAIL':>6}{'ok' if r['drift_ok'] else 'DRIFT':>7}")
        print("\n  Judge the saved panels by eye: is the corrected similarity upright, are the sleeves natural, is the torso covered, are the shoulders exposed?")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="C2: corrected-underarm baseline (similarity vs 4-handle, original vs corrected avatar underarms).")
    ap.add_argument("--garment-ids", nargs="*", default=[GOLDEN_GARMENT_ID, SECOND_GARMENT_ID, WHITE_TEE_ID])
    args = ap.parse_args()
    main(args.garment_ids)