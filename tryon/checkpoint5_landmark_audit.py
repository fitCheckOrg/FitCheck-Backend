"""
tryon/checkpoint5_landmark_audit.py

READ-ONLY LANDMARK AUDIT. No solver, mesh, handle or deformation code is run. It looks at the landmarks the pipeline
produces, on the avatar and on the garments, to answer three questions that the last experiments left open.

  Q1  WHERE DOES THE ~9 DEG TILT COME FROM?  Every underarm-similarity render is tilted; the rotation is
      (avatar underarm-line angle) - (garment underarm-line angle). Is the avatar's underarm asymmetry (16 px, -9.0 deg
      while its shoulders are level) real body geometry or detector noise? An INDEPENDENT estimate is taken from the
      avatar silhouette alone: the row where each arm first separates from the torso (the armpit apex) and the lean of
      the torso centreline. The same silhouette-centreline method gives each garment's own axis tilt.
  Q2  WHAT ARE THE GARMENT 'SHOULDER' POINTS?  Where does each shoulder point sit relative to its sleeve tip (the
      silhouette's extreme-x point) and to the silhouette boundary?
  Q3  IS A NECKLINE CANDIDATE DETECTABLE ON THE GARMENT SIDE?  A top-centre point and a neck-notch depth are measured
      on every garment (including the shoulderless one) and compared across garments. The AVATAR-side target for any
      such point is NOT evaluated here.

Rotation bookkeeping (image coordinates, y down; same convention as the similarity solver):
      theta = angle(avatar line) - angle(garment line)            (underarm line: what the pipeline uses today)
      theta_axis = phi_garment - phi_avatar                       (phi = lean of the silhouette centreline from vertical)
If both axes are upright theta_axis ~ 0, so a large theta_underarm is then an artefact of the underarm LINES.

REGISTERED INTERPRETATION RULES (fixed before any data; the 3 deg and 20 px thresholds are mine, informational):
  R1  avatar tilt:
        apex line within 3 deg of the pipeline underarm line         -> the asymmetry is in the silhouette itself
                                                                        (arm geometry), not detector noise
        apex line level (|angle| <= 3) while the pipeline line is not -> the pipeline's underarm LANDMARKS are the source
        otherwise / no gap found                                      -> inconclusive
        torso centreline |phi_avatar| <= 3 deg                        -> the torso is upright
  R2  garment contribution to the rotation: |garment underarm-line angle| > 3 deg means garment-side detection adds tilt.
  R3  shoulder semantics: both shoulder points within 20 px of their sleeve tips AND shoulder span / tip span >= 0.90
        -> the 'shoulder' points are the outer sleeve corners, not shoulder joints.
  R4  neckline candidate: values and spreads are printed; no verdict (detectability only).

Run: python -m tryon.checkpoint5_landmark_audit [--garment-ids ID [ID ...]]
Default ids: golden garment, 831098a3-..., and the white tee 7e16734e-... Output: console + 2 images.
"""

import argparse
import os
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw

import tryon.garment_representation as garment_module
from tryon.avatar_representation import build_avatar_representation
from tryon.piecewise_continuous_test import solve_similarity_2point
from tryon.checkpoint5_ARAP0_deformation_field import GOLDEN_GARMENT_ID, AVATAR_USER_ID
from tryon.checkpoint5_ARAP1_R2 import pair_by_x
from tryon.checkpoint5_S3_underarm import tolerant_shoulders, WHITE_TEE_ID, SECOND_GARMENT_ID

OUTPUT_DIR = "outputs"
TILT_TOL = 3.0       # degrees
NEAR_TIP_PX = 20.0   # pixels


# ----------------------------------------------------------------------------- geometry helpers
def ang(p_small, p_large):
    return float(np.degrees(np.arctan2(p_large[1] - p_small[1], p_large[0] - p_small[0])))


def silhouette(img):
    arr = np.array(img.convert("RGBA"))
    a = arr[:, :, 3]
    if a.min() < 128 <= a.max():
        return a >= 128, "alpha channel"
    return arr[:, :, :3].astype(int).sum(axis=2) > 36, "non-black pixels"


def runs_of(row):
    idx = np.flatnonzero(row)
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[splits + 1]]
    ends = np.r_[idx[splits], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def fit_axis(centres):
    """centres: list of (y, x). Returns (phi_deg, n, resid_std) where phi = atan(dx/dy) in degrees."""
    if len(centres) < 8:
        return None
    ys = np.array([c[0] for c in centres], float)
    xs = np.array([c[1] for c in centres], float)
    slope, icpt = np.polyfit(ys, xs, 1)
    resid = float(np.std(xs - (slope * ys + icpt)))
    return float(np.degrees(np.arctan(slope))), len(centres), resid


# ----------------------------------------------------------------------------- avatar
def avatar_audit(avatar):
    F, how = silhouette(avatar.image)
    H, W = F.shape
    sh = pair_by_x(avatar.left_shoulder, avatar.right_shoulder)
    ua = pair_by_x(avatar.left_underarm, avatar.right_underarm)
    x_c = float(np.mean([p[0] for p in sh]))
    y_sh = float(np.mean([p[1] for p in sh]))
    y_end = min(H - 1, int(max(p[1] for p in ua)) + 100)
    sep = {"small-x": [], "large-x": []}
    torso_rows = {}
    for y in range(int(y_sh), y_end + 1):
        runs = runs_of(F[y])
        if not runs:
            continue
        inside = [r for r in runs if r[0] <= x_c <= r[1]]
        torso = inside[0] if inside else min(runs, key=lambda r: abs((r[0] + r[1]) / 2 - x_c))
        left = [r for r in runs if r[1] < torso[0] - 1 and (r[1] - r[0] + 1) >= 3]
        right = [r for r in runs if r[0] > torso[1] + 1 and (r[1] - r[0] + 1) >= 3]
        torso_rows[y] = (torso, bool(left), bool(right))
        sep["small-x"].append((y, bool(left), ((max(r[1] for r in left) + torso[0]) / 2.0) if left else None))
        sep["large-x"].append((y, bool(right), ((torso[1] + min(r[0] for r in right)) / 2.0) if right else None))
    apex = {}
    for side, lst in sep.items():
        apex[side] = None
        for k in range(len(lst) - 2):
            if lst[k][1] and lst[k + 1][1] and lst[k + 2][1]:
                apex[side] = (float(lst[k][2]), float(lst[k][0]))
                break
    out = dict(mask_source=how, apex=apex, shoulders=sh, underarms=ua, F=F)
    out["ang_shoulder"] = ang(sh[0], sh[1])
    out["ang_underarm"] = ang(ua[0], ua[1])
    out["ang_apex"] = ang(apex["small-x"], apex["large-x"]) if apex["small-x"] and apex["large-x"] else None
    out["axis"] = None
    if apex["small-x"] and apex["large-x"]:
        y0 = int(max(apex["small-x"][1], apex["large-x"][1])) + 10
        centres = [(y, (torso_rows[y][0][0] + torso_rows[y][0][1]) / 2.0) for y in range(y0, y0 + 130)
                   if y in torso_rows and torso_rows[y][1] and torso_rows[y][2]]
        out["axis"] = fit_axis(centres)
    return out


# ----------------------------------------------------------------------------- garment
def garment_audit(garment):
    mask = np.asarray(garment.mask).astype(bool)
    H, W = mask.shape
    ys, xs = np.where(mask)
    ua = pair_by_x(garment.left_underarm, garment.right_underarm)
    tips = (( float(xs.min()), float(ys[xs == xs.min()].mean())), (float(xs.max()), float(ys[xs == xs.max()].mean())))
    y_top = int(ys.min())
    sil_h = int(ys.max() - ys.min() + 1)
    sil_w = float(xs.max() - xs.min() + 1)
    band = max(2, int(0.03 * sil_h))
    top_xs = xs[ys <= y_top + 1]
    top_center = (float(np.median(top_xs)), float(y_top))
    top_center_band = (float(np.median(xs[ys <= y_top + band])), float(y_top))
    ua_mid = ((ua[0][0] + ua[1][0]) / 2.0, (ua[0][1] + ua[1][1]) / 2.0)
    ua_span = float(np.hypot(ua[1][0] - ua[0][0], ua[1][1] - ua[0][1]))

    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    hm = np.zeros((H, W), np.uint8)
    cv2.fillPoly(hm, [hull], 1)
    cx = (xs.min() + xs.max()) / 2.0
    band_lo, band_hi = int(cx - 0.2 * sil_w), int(cx + 0.2 * sil_w)
    best = (0.0, None)
    for x in range(max(0, band_lo), min(W, band_hi + 1)):
        mc = np.flatnonzero(mask[:, x])
        hc = np.flatnonzero(hm[:, x])
        if mc.size and hc.size:
            d = float(mc[0] - hc[0])
            if d > best[0]:
                best = (d, (float(x), float(mc[0])))
    notch_depth, notch_pt = best

    y0 = int(ua_mid[1]) + 10
    centres = []
    for y in range(y0, int(ys.max()) - 10):
        r = np.flatnonzero(mask[y])
        if r.size:
            centres.append((y, (r[0] + r[-1]) / 2.0))
    out = dict(mask=mask, contour=contour, ua=ua, tips=tips, top_center=top_center, top_center_band=top_center_band,
               ua_mid=ua_mid, ua_span=ua_span, notch_depth=notch_depth, notch_pt=notch_pt, sil_w=sil_w, sil_h=sil_h,
               ang_underarm=ang(ua[0], ua[1]), ang_tips=ang(tips[0], tips[1]), axis=fit_axis(centres),
               has_shoulders=bool(garment.shoulders_available))
    out["top_x_off_pct"] = 100.0 * (top_center[0] - ua_mid[0]) / ua_span
    out["top_rise_pct"] = 100.0 * (ua_mid[1] - top_center[1]) / ua_span
    out["top_band_dx"] = top_center_band[0] - top_center[0]
    if garment.shoulders_available:
        sh = pair_by_x(garment.left_shoulder, garment.right_shoulder)
        out["sh"] = sh
        out["sh_to_tip"] = [float(np.hypot(sh[k][0] - tips[k][0], sh[k][1] - tips[k][1])) for k in (0, 1)]
        out["sh_to_boundary"] = [abs(float(cv2.pointPolygonTest(contour, (float(sh[k][0]), float(sh[k][1])), True))) for k in (0, 1)]
        out["sh_span"] = float(np.hypot(sh[1][0] - sh[0][0], sh[1][1] - sh[0][1]))
        out["tip_span"] = float(np.hypot(tips[1][0] - tips[0][0], tips[1][1] - tips[0][1]))
        out["ang_shoulder"] = ang(sh[0], sh[1])
    return out


# ----------------------------------------------------------------------------- drawing
def _dot(d, p, r, color, width=2):
    d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], outline=color, width=width)


def draw_avatar(avatar, A):
    x0, y0, x1, y1, k = 15, 100, 208, 350, 3
    arr = np.array(avatar.image.convert("RGBA"))
    base = Image.fromarray(arr[:, :, :3]).crop((x0, y0, x1, y1)).resize(((x1 - x0) * k, (y1 - y0) * k), Image.LANCZOS)
    d = ImageDraw.Draw(base)
    S = lambda p: ((p[0] - x0) * k, (p[1] - y0) * k)
    sh, ua = A["shoulders"], A["underarms"]
    for p in sh:
        _dot(d, S(p), 9, (60, 255, 60), 3)
    d.line([S(sh[0]), S(sh[1])], fill=(60, 255, 60), width=2)
    for p in ua:
        _dot(d, S(p), 9, (0, 220, 255), 3)
    d.line([S(ua[0]), S(ua[1])], fill=(0, 220, 255), width=2)
    for side in ("small-x", "large-x"):
        if A["apex"][side]:
            p = S(A["apex"][side]); d.line([p[0] - 9, p[1] - 9, p[0] + 9, p[1] + 9], fill=(255, 60, 255), width=3); d.line([p[0] - 9, p[1] + 9, p[0] + 9, p[1] - 9], fill=(255, 60, 255), width=3)
    if A["apex"]["small-x"] and A["apex"]["large-x"]:
        d.line([S(A["apex"]["small-x"]), S(A["apex"]["large-x"])], fill=(255, 60, 255), width=2)
    if A["axis"]:
        ys_ = [int(max(A["apex"]["small-x"][1], A["apex"]["large-x"][1])) + 10, int(max(A["apex"]["small-x"][1], A["apex"]["large-x"][1])) + 140]
        xc = float(np.mean([p[0] for p in sh]))
        slope = np.tan(np.radians(A["axis"][0]))
        # centreline through the mean torso centre implied by the fit is unavailable here; draw through the shoulder mid-x
        d.line([S((xc + slope * (ys_[0] - 163), ys_[0])), S((xc + slope * (ys_[1] - 163), ys_[1]))], fill=(255, 255, 0), width=2)
    d.text((6, 6), "AVATAR  green=shoulders  cyan=pipeline underarms  magenta X=silhouette armpit apex  yellow=torso axis", fill=(255, 255, 255))
    return base


def draw_garments(items):
    panels = []
    for gid, G in items:
        gm = G["_garment"]
        W, H = G["mask"].shape[1], G["mask"].shape[0]
        base = Image.new("RGB", (W, H), (205, 205, 205))
        img = gm.image.convert("RGBA")
        base.paste(img.convert("RGB"), (0, 0), img.split()[3])
        d = ImageDraw.Draw(base)
        pts = [tuple(map(float, p[0])) for p in G["contour"]]
        d.line(pts + [pts[0]], fill=(255, 220, 0), width=2)
        for p in G["ua"]:
            _dot(d, p, 9, (0, 190, 255), 3)
        d.line([G["ua"][0], G["ua"][1]], fill=(0, 190, 255), width=2)
        for p in G["tips"]:
            _dot(d, p, 6, (255, 40, 255), 3)
        _dot(d, G["top_center"], 8, (255, 140, 0), 3)
        if G["notch_pt"] is not None and G["notch_depth"] >= 0.03 * G["sil_w"]:
            _dot(d, G["notch_pt"], 8, (255, 0, 0), 3)
        if G["has_shoulders"]:
            for k, p in enumerate(G["sh"]):
                _dot(d, p, 9, (60, 255, 60), 3)
                d.line([p, G["tips"][k]], fill=(60, 255, 60), width=2)
        d.text((6, 6), f"{gid[:8]}  yellow=outline cyan=underarms magenta=sleeve tips orange=top-centre"
                       + ("  green=shoulders" if G["has_shoulders"] else "  (no shoulders)"), fill=(0, 0, 0))
        panels.append(base)
    Hmax = max(p.size[1] for p in panels)
    Wtot = sum(p.size[0] for p in panels) + 10 * (len(panels) - 1)
    sheet = Image.new("RGB", (Wtot, Hmax), (40, 40, 40))
    x = 0
    for p in panels:
        sheet.paste(p, (x, 0)); x += p.size[0] + 10
    return sheet


# ----------------------------------------------------------------------------- main
def main(ids):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(__doc__)
    print("Building the avatar...")
    avatar = build_avatar_representation()
    bar = "=" * 70

    print(f"\n{bar}\nAVATAR ATTRIBUTES AVAILABLE (so we know what else the pipeline holds)\n{bar}")
    for name in sorted(n for n in dir(avatar) if not n.startswith("_")):
        try:
            v = getattr(avatar, name)
        except Exception as exc:
            v = f"<raises {type(exc).__name__}>"
        if callable(v):
            continue
        desc = type(v).__name__
        if hasattr(v, "shape"):
            desc += f" shape={tuple(v.shape)}"
        elif hasattr(v, "size") and not isinstance(v, (int, float, str)):
            desc += f" size={v.size}"
        elif isinstance(v, (tuple, list)) and len(v) <= 4:
            desc += f" {v}"
        print(f"  {name:<28} {desc}")

    A = avatar_audit(avatar)
    print(f"\n{bar}\nQ1  AVATAR: TILT, FROM THE PIPELINE LANDMARKS AND FROM THE SILHOUETTE ALONE\n{bar}")
    print(f"  silhouette source: {A['mask_source']}")
    sh, ua = A["shoulders"], A["underarms"]
    print(f"  pipeline shoulders  small-x=({sh[0][0]:.1f},{sh[0][1]:.1f}) large-x=({sh[1][0]:.1f},{sh[1][1]:.1f})  vertical diff={sh[1][1] - sh[0][1]:+.1f}px  line angle={A['ang_shoulder']:+.2f} deg")
    print(f"  pipeline underarms  small-x=({ua[0][0]:.1f},{ua[0][1]:.1f}) large-x=({ua[1][0]:.1f},{ua[1][1]:.1f})  vertical diff={ua[1][1] - ua[0][1]:+.1f}px  line angle={A['ang_underarm']:+.2f} deg")
    for side in ("small-x", "large-x"):
        ap = A["apex"][side]
        k = 0 if side == "small-x" else 1
        if ap:
            print(f"  silhouette armpit apex, {side}: ({ap[0]:.1f},{ap[1]:.0f})   pipeline underarm landmark is {np.hypot(ap[0] - ua[k][0], ap[1] - ua[k][1]):.1f}px from it (dy={ua[k][1] - ap[1]:+.1f}px)")
        else:
            print(f"  silhouette armpit apex, {side}: NOT FOUND (the arm never separates from the torso in the scanned rows)")
    if A["ang_apex"] is not None:
        print(f"  silhouette apex line: vertical diff={A['apex']['large-x'][1] - A['apex']['small-x'][1]:+.1f}px  angle={A['ang_apex']:+.2f} deg")
    if A["axis"]:
        print(f"  torso centreline lean phi_avatar = {A['axis'][0]:+.2f} deg  ({A['axis'][1]} rows, residual std {A['axis'][2]:.2f}px)")
    else:
        print("  torso centreline lean: unavailable")
    ap_deg, pl_deg = A["ang_apex"], A["ang_underarm"]
    if ap_deg is None:
        r1 = "inconclusive (no armpit apex on both sides)"
    elif abs(ap_deg - pl_deg) <= TILT_TOL:
        r1 = "the underarm asymmetry is in the SILHOUETTE itself (arm geometry), not detector noise"
    elif abs(ap_deg) <= TILT_TOL and abs(pl_deg) > TILT_TOL:
        r1 = "the silhouette is LEVEL while the pipeline underarm line is not: the pipeline's underarm LANDMARKS are the tilt source"
    else:
        r1 = "inconclusive (the silhouette apex line differs from both level and the pipeline line)"
    print(f"  R1 reading: {r1}")
    if A["axis"]:
        print(f"  R1 torso: {'UPRIGHT' if abs(A['axis'][0]) <= TILT_TOL else 'LEANING'} (|phi_avatar| {'<=' if abs(A['axis'][0]) <= TILT_TOL else '>'} {TILT_TOL} deg)")

    items = []
    results = {}
    for gid in ids:
        print(f"\n{bar}\nGARMENT {gid}\n{bar}")
        try:
            with tolerant_shoulders():
                gm = garment_module.build_garment_representation(gid, AVATAR_USER_ID)
        except Exception as exc:
            print(f"NOT TESTABLE: garment representation failed: {type(exc).__name__}: {exc}")
            continue
        G = garment_audit(gm)
        G["_garment"] = gm
        items.append((gid, G))
        results[gid] = G
        g_ua = G["ua"]
        print(f"  size={gm.image.size} silhouette {G['sil_w']:.0f}x{G['sil_h']} px; shoulders_available={G['has_shoulders']}")
        print(f"  underarms small-x=({g_ua[0][0]:.0f},{g_ua[0][1]:.0f}) large-x=({g_ua[1][0]:.0f},{g_ua[1][1]:.0f})  vertical diff={g_ua[1][1] - g_ua[0][1]:+.0f}px  span={G['ua_span']:.0f}px  line angle={G['ang_underarm']:+.2f} deg")
        print(f"  sleeve tips (extreme x) small-x=({G['tips'][0][0]:.0f},{G['tips'][0][1]:.0f}) large-x=({G['tips'][1][0]:.0f},{G['tips'][1][1]:.0f})  tip-line angle={G['ang_tips']:+.2f} deg")
        theta_ua = A["ang_underarm"] - G["ang_underarm"]
        T = np.array(solve_similarity_2point(list(g_ua), list(ua)))
        T_rot = float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))
        print(f"  ROTATION DECOMPOSITION (underarm-line similarity): avatar line {A['ang_underarm']:+.2f} - garment line {G['ang_underarm']:+.2f} = {theta_ua:+.2f} deg"
              f"   (solver's rotation {T_rot:+.2f} deg -> {'consistent' if abs(theta_ua - T_rot) < 0.01 else 'MISMATCH'})")
        print(f"    R2: garment-side contribution |{G['ang_underarm']:+.2f}| deg is {'ABOVE' if abs(G['ang_underarm']) > TILT_TOL else 'within'} {TILT_TOL} deg"
              f" -> garment underarm detection {'adds tilt' if abs(G['ang_underarm']) > TILT_TOL else 'is not a tilt source'}")
        if G["axis"]:
            print(f"  garment symmetry-axis lean phi_garment = {G['axis'][0]:+.2f} deg ({G['axis'][1]} rows, residual std {G['axis'][2]:.2f}px)")
            if A["axis"]:
                th_axis = G["axis"][0] - A["axis"][0]
                print(f"  AXIS-IMPLIED rotation (phi_garment - phi_avatar) = {th_axis:+.2f} deg   vs underarm-line rotation {theta_ua:+.2f} deg   (difference {theta_ua - th_axis:+.2f} deg)")
        if G["has_shoulders"]:
            sh_g = G["sh"]
            th_sh = A["ang_shoulder"] - G["ang_shoulder"]
            print(f"  shoulders small-x=({sh_g[0][0]:.0f},{sh_g[0][1]:.0f}) large-x=({sh_g[1][0]:.0f},{sh_g[1][1]:.0f})  vertical diff={sh_g[1][1] - sh_g[0][1]:+.0f}px  line angle={G['ang_shoulder']:+.2f} deg"
                  f"  -> shoulder-line rotation would be {th_sh:+.2f} deg")
            near = all(v <= NEAR_TIP_PX for v in G["sh_to_tip"]) and (G["sh_span"] / G["tip_span"] >= 0.90)
            print(f"  Q2 shoulder point -> its sleeve tip: {G['sh_to_tip'][0]:.1f}px / {G['sh_to_tip'][1]:.1f}px;  distance to the silhouette boundary: {G['sh_to_boundary'][0]:.1f}px / {G['sh_to_boundary'][1]:.1f}px")
            print(f"    shoulder span {G['sh_span']:.0f}px vs sleeve-tip span {G['tip_span']:.0f}px (ratio {G['sh_span'] / G['tip_span']:.3f}); avatar shoulder span {np.hypot(sh[1][0]-sh[0][0], sh[1][1]-sh[0][1]):.0f}px")
            print(f"    R3 reading: " + ("the 'shoulder' points ARE the outer sleeve corners, not shoulder joints" if near else "the shoulder points are NOT simply the sleeve tips"))
        else:
            print("  Q2: no shoulder landmarks on this garment (nothing to place relative to the sleeve tips)")
        print(f"  Q3 neckline candidates (garment side only):")
        print(f"    top-centre point ({G['top_center'][0]:.0f},{G['top_center'][1]:.0f}); offset from the underarm midpoint x = {G['top_x_off_pct']:+.1f}% of the underarm span; "
              f"height above the underarm line = {G['top_rise_pct']:.1f}% of the underarm span; stability: using a 3%-height band moves it {G['top_band_dx']:+.1f}px in x")
        nd_flag = "NOTCH PRESENT" if G["notch_depth"] >= 0.03 * G["sil_w"] else "no notch (solid top edge)"
        print(f"    neck-notch depth in the central 40% of the width: {G['notch_depth']:.1f}px ({100 * G['notch_depth'] / G['sil_w']:.1f}% of width) -> {nd_flag}"
              + (f" at ({G['notch_pt'][0]:.0f},{G['notch_pt'][1]:.0f})" if G["notch_pt"] is not None and G['notch_depth'] > 0 else ""))

    if results:
        print(f"\n{bar}\nCROSS-GARMENT SUMMARY\n{bar}")
        labels = list(results)
        def line(name, f, w=14):
            print(f"  {name:<52}" + "".join(f"{f(results[g]):>{w}}" for g in labels))
        print(f"  {'garment':<52}" + "".join(f"{g[:8]:>14}" for g in labels))
        line("garment underarm-line angle (deg)", lambda G: f"{G['ang_underarm']:+.2f}")
        line("garment sleeve-tip-line angle (deg)", lambda G: f"{G['ang_tips']:+.2f}")
        line("garment symmetry-axis lean (deg)", lambda G: (f"{G['axis'][0]:+.2f}" if G["axis"] else "n/a"))
        line("rotation from the underarm lines (deg)", lambda G: f"{A['ang_underarm'] - G['ang_underarm']:+.2f}")
        line("rotation implied by the symmetry axes (deg)", lambda G: (f"{G['axis'][0] - A['axis'][0]:+.2f}" if (G["axis"] and A["axis"]) else "n/a"))
        line("shoulder point -> sleeve tip (px, worst side)", lambda G: (f"{max(G['sh_to_tip']):.1f}" if G["has_shoulders"] else "no shoulders"))
        line("shoulder span / sleeve-tip span", lambda G: (f"{G['sh_span'] / G['tip_span']:.3f}" if G["has_shoulders"] else "no shoulders"))
        line("top-centre x offset (% of underarm span)", lambda G: f"{G['top_x_off_pct']:+.1f}")
        line("top-centre height above underarms (% of span)", lambda G: f"{G['top_rise_pct']:.1f}")
        line("neck-notch depth (% of width)", lambda G: f"{100 * G['notch_depth'] / G['sil_w']:.1f}")
        tx = [G["top_x_off_pct"] for G in results.values()]
        tr = [G["top_rise_pct"] for G in results.values()]
        print(f"  top-centre candidate spread across garments: x offset {max(tx) - min(tx):.1f} points, height {max(tr) - min(tr):.1f} points (of the underarm span)")
        print(f"  (the avatar-side target of any neckline point is NOT evaluated here)")

    av_img = draw_avatar(avatar, A)
    av_path = f"{OUTPUT_DIR}/landmark_audit_avatar.png"
    av_img.save(av_path)
    print(f"\nSaved: {av_path}  (avatar crop, 3x)")
    if items:
        g_img = draw_garments(items)
        g_path = f"{OUTPUT_DIR}/landmark_audit_garments.png"
        g_img.save(g_path)
        print(f"Saved: {g_path}  ({len(items)} garments side by side)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Read-only landmark audit of the avatar and garments.")
    ap.add_argument("--garment-ids", nargs="*", default=[GOLDEN_GARMENT_ID, SECOND_GARMENT_ID, WHITE_TEE_ID])
    args = ap.parse_args()
    main(args.garment_ids)