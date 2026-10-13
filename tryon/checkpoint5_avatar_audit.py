"""
tryon/checkpoint5_avatar_audit.py

READ-ONLY AVATAR AUDIT FOR A GIVEN AVATAR USER. No solver, mesh, handle or deformation code is run, nothing is
uploaded, nothing is written to the backend. It generalises the avatar half of checkpoint5_landmark_audit.py so the
corrected-underarm (C2) finding can be checked on avatars other than the golden one.

What changes relative to the golden-only audit
  * takes one or more avatar USER IDS (build_avatar_representation(avatar_user_id=...)); no id = the golden avatar
  * never touches rep.left_underarm / rep.right_underarm without a guard (they raise on an avatar whose detector fails);
    the detector is called DIRECTLY (find_semantic_underarm) and None is a legitimate, reported result
  * every pixel window is a multiple of the avatar's own shoulder span, not a constant tuned to the golden image
  * separates the two ways an avatar can break the check:
        MASK FAILURE        the SCHP torso/arm masks have a hole, the foreground silhouette is intact
                            -> the pipeline detector fails, the silhouette apex is still valid
        FOREGROUND FAILURE  the foreground itself is damaged (fragmented, holes on the centreline, dark skin lost
                            against a black background) -> the silhouette apex is NOT trustworthy either

REGISTERED INTERPRETATION RULES (fixed before any data; thresholds are mine and informational)
  A1  apex found on both sides                          -> the independent underarm estimate exists for this avatar
  A2  silhouette apex line within 3 deg of level        -> the avatar's own underarms are level (informational)
  A3  per side: detector result vs apex
        detector returned None                          -> DETECTOR FAILED (no landmark to compare)
        |dy| <= 3 px                                    -> detector agrees with the silhouette
        else                                            -> detector DISAGREES (the C2 situation: wrong underarm Y)
  F1  foreground intact = in the torso band, >= 97 % of foreground pixels belong to ONE connected component,
      centreline missing <= 2 % of torso rows, enclosed holes <= 1 % of the CENTRAL torso strip
      (|x - centre| <= 0.25 shoulder span; closed arm/body gaps elsewhere are not damage)
  M1  mask hole = more than 15 % of the central foreground in the torso band is covered by neither the torso nor the
      upper-arm mask
  V   verdict:  not F1                -> FOREGROUND FAILURE, apex not trustworthy
                F1 and M1             -> MASK FAILURE, foreground intact, apex valid
                F1 and not M1         -> masks and foreground both fine
  A4  apex depth = (apex y - shoulder y) / shoulder span must lie in 0.20..0.80 (golden: 0.50). Outside that the
      'apex' is probably a hand/elbow gap or a damaged silhouette, not an armpit -> not usable.
  D   golden drift guard (only when no avatar id is given): apex Y within 1.5 px of the recorded 223 / 223 (C2 kept the
      pipeline X and replaced only Y, so the apex X is NOT part of the record; it is printed against the pipeline X
      for information).
  A5  apex stability (no-alpha images only): the apex is recomputed with the foreground threshold lowered from
      sum>36 to sum>6 (largest component only). If either apex Y moves by more than 3 px the sum>36 apex is probably a
      DARK CREASE (dark skin pixels between arm and chest counted as background), not the armpit -> UNSTABLE, and the
      low-threshold apex is the one to trust (if it is plausible and clean). Found on avatar #2: the global-area check
      (x1.01) cannot see a 2 px crease, this one can.
  NOTE find_semantic_underarm returns (row, [adjacent rows ...]) - a ROW only, no x. The detector comparison (A3) is
      therefore on Y only; 'left' is taken as the small-x side (golden: left row 229 = pipeline left landmark (61,229)).

Run (from the backend root, same place as the other checkpoint5 scripts):
    python -m tryon.checkpoint5_avatar_audit                                   # golden avatar (regression)
    python -m tryon.checkpoint5_avatar_audit --avatar-user-id f4b5f129-db34-40e5-8423-d1b256a15578
    python -m tryon.checkpoint5_avatar_audit --avatar-user-id golden f4b5f129-... <more ids>
The token 'golden' means "the default avatar". Output: console + outputs/avatar_audit_<label>.png and .json
"""

import argparse
import json
import os
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw

from tryon.avatar_representation import build_avatar_representation
from tryon.avatar_torso_landmarks_test import find_semantic_underarm

OUTPUT_DIR = "outputs"
TILT_TOL = 3.0             # degrees
AGREE_PX = 3.0             # pixels
FG_COMPONENT_MIN = 0.97
CENTRELINE_MISSING_MAX = 0.02
HOLES_MAX = 0.01
MASK_UNCOVERED_MAX = 0.15
GOLDEN_APEX_Y = (223.0, 223.0)
GOLDEN_APEX_TOL = 1.5
DEPTH_MIN, DEPTH_MAX = 0.20, 0.80


# ----------------------------------------------------------------------------- helpers
def ang(p_small, p_large):
    return float(np.degrees(np.arctan2(p_large[1] - p_small[1], p_large[0] - p_small[0])))


def guarded(fn, default=None):
    """Return (value, error-string). Used for every pipeline attribute that can raise."""
    try:
        return fn(), None
    except Exception as exc:                      # noqa: BLE001 - the whole point is to survive any detector failure
        return default, f"{type(exc).__name__}: {exc}"


def foreground(img):
    """Same rule as checkpoint5_landmark_audit.silhouette, plus the source."""
    arr = np.array(img.convert("RGBA"))
    a = arr[:, :, 3]
    if a.min() < 128 <= a.max():
        return a >= 128, "alpha channel", arr
    return arr[:, :, :3].astype(int).sum(axis=2) > 36, "non-black pixels (NO usable alpha)", arr


def runs_of(row):
    idx = np.flatnonzero(row)
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[splits + 1]]
    ends = np.r_[idx[splits], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def fit_axis(centres):
    if len(centres) < 8:
        return None
    ys = np.array([c[0] for c in centres], float)
    xs = np.array([c[1] for c in centres], float)
    slope, icpt = np.polyfit(ys, xs, 1)
    return float(np.degrees(np.arctan(slope))), len(centres), float(np.std(xs - (slope * ys + icpt)))


def pt(p):
    """Normalise a detector/landmark return value to (x, y). None -> None. An unfamiliar shape raises ValueError
    (the caller reports it with the raw value instead of crashing)."""
    if p is None:
        return None
    if isinstance(p, dict) and "x" in p and "y" in p:
        return (float(p["x"]), float(p["y"]))
    try:
        a = np.asarray(p, dtype=float).ravel()
        if a.size == 2 and np.all(np.isfinite(a)):
            return (float(a[0]), float(a[1]))
    except (TypeError, ValueError):
        pass
    if isinstance(p, (tuple, list)) and len(p) >= 1:                  # (point, extra, ...) shapes
        try:
            a = np.asarray(p[0], dtype=float).ravel()
            if a.size == 2 and np.all(np.isfinite(a)):
                return (float(a[0]), float(a[1]))
        except (TypeError, ValueError):
            pass
    raise ValueError("unrecognised point shape: " + type(p).__name__)


def parse_detector(v):
    """find_semantic_underarm returns (row, [adjacent rows ...]) on the real pipeline: a ROW, no x.
    Returns dict(y, x, rows) or None. Falls back to an (x, y) point. Raises ValueError for any other shape."""
    if v is None:
        return None
    if isinstance(v, (tuple, list)) and len(v) == 2 and v[0] is None:          # (None, []) = nothing found
        return None
    if isinstance(v, (tuple, list)) and len(v) == 2 and np.isscalar(v[0]) and isinstance(v[1], (list, tuple, np.ndarray)):
        rows = np.asarray(v[1], dtype=float).ravel()
        r = (int(rows.min()), int(rows.max()), int(rows.size)) if rows.size else None
        return dict(y=float(v[0]), x=None, rows=r)
    x, y = pt(v)
    return dict(y=y, x=x, rows=None)


def short(v, n=160):
    r = repr(v)
    return r if len(r) <= n else r[:n] + "...(truncated)"


# ----------------------------------------------------------------------------- foreground / mask diagnostics
def foreground_diagnostics(F, how, arr, x_c, y_sh, span):
    H, W = F.shape
    area = int(F.sum())
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    fl = lum[F]
    out = dict(size=(W, H), source=how, area=area, area_frac=area / float(W * H))
    out["mean_lum"] = float(fl.mean()) if fl.size else float("nan")
    out["near_black_frac"] = float((fl < 20).mean()) if fl.size else float("nan")
    out["contrast_vs_black"] = out["mean_lum"]
    out["contrast_vs_white"] = 255.0 - out["mean_lum"]

    if "NO usable alpha" in how:
        # how much does the foreground grow if the 'non-black' threshold is lowered from sum>36 to sum>6?
        alt = arr[:, :, :3].astype(int).sum(axis=2) > 6
        out["alt_area_ratio"] = float(alt.sum() / max(1, area))
    else:
        out["alt_area_ratio"] = None
    n, lab, stats, _ = cv2.connectedComponentsWithStats(F.astype(np.uint8), connectivity=8)
    areas = sorted((int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)), reverse=True)
    out["components"] = len(areas)
    out["largest_frac"] = (areas[0] / float(area)) if areas and area else 0.0

    # enclosed holes: background pixels not connected to the image border
    inv = (~F).astype(np.uint8)
    ff = inv.copy()
    pad = np.pad(ff, 1, constant_values=1)
    mask = np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8)
    cv2.floodFill(pad, mask, (0, 0), 2)
    hole_mask = (pad[1:-1, 1:-1] == 1)
    holes = int(hole_mask.sum())
    out["holes_frac"] = holes / float(area) if area else 0.0          # whole figure, informational only

    # pixels inside the convex hull of the foreground that are NOT foreground (dark skin lost = a big gap here)
    cnts, _ = cv2.findContours(F.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts:
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
        hm = np.zeros((H, W), np.uint8)
        cv2.fillPoly(hm, [hull], 1)
        out["hull_gap_frac"] = float(((hm > 0) & ~F).sum() / max(1, hm.sum()))
    else:
        out["hull_gap_frac"] = float("nan")

    # centreline presence over the torso band
    y0, y1 = int(max(0, y_sh)), int(min(H - 1, y_sh + 1.1 * span))
    band_labels = lab[y0:y1 + 1][F[y0:y1 + 1]]
    if band_labels.size:
        counts = np.bincount(band_labels)
        out["band_main_frac"] = float(counts.max() / band_labels.size)
    else:
        out["band_main_frac"] = 0.0
    xi = int(round(min(max(x_c, 0), W - 1)))
    rows = list(range(y0, y1 + 1))
    miss = sum(1 for y in rows if not F[y, xi])
    out["centreline_missing"] = miss / float(max(1, len(rows)))
    out["band"] = (y0, y1)
    # enclosed holes that matter: inside the central strip of the torso band (the closed gaps between an arm and the
    # body of an arms-away pose are NOT damage and are excluded by looking only at the centre)
    strip = np.zeros((H, W), bool)
    xl, xr = int(max(0, x_c - 0.25 * span)), int(min(W - 1, x_c + 0.25 * span))
    strip[y0:y1 + 1, xl:xr + 1] = True
    strip_area = int(strip.sum())
    out["holes_central_frac"] = float((hole_mask & strip).sum() / max(1, strip_area))
    out["fg_intact"] = bool(out["band_main_frac"] >= FG_COMPONENT_MIN and out["centreline_missing"] <= CENTRELINE_MISSING_MAX
                            and out["holes_central_frac"] <= HOLES_MAX)
    return out


def mask_diagnostics(F, torso, arms, x_c, span, band):
    H, W = F.shape
    y0, y1 = band
    out = dict(torso_area=int(torso.sum()), arms_area=int(arms.sum()),
               torso_frac_of_fg=float(torso.sum() / max(1, F.sum())), arms_frac_of_fg=float(arms.sum() / max(1, F.sum())))
    xs = np.arange(W)[None, :]
    central = F & (np.abs(xs - x_c) <= 0.25 * span)
    central[:y0] = False
    central[y1 + 1:] = False
    covered = (torso | arms) & central
    n = int(central.sum())
    out["central_fg_px"] = n
    out["uncovered_frac"] = float(1.0 - covered.sum() / n) if n else float("nan")
    unc = central & ~(torso | arms)
    if unc.any():
        ys_, xs_ = np.where(unc)
        out["uncovered_bbox"] = (int(xs_.min()), int(ys_.min()), int(xs_.max()), int(ys_.max()))
    else:
        out["uncovered_bbox"] = None
    out["mask_hole"] = bool(n and out["uncovered_frac"] > MASK_UNCOVERED_MAX)
    return out


# ----------------------------------------------------------------------------- silhouette apex (window scaled to span)
def silhouette_apex(F, x_c, y_sh, span):
    H, W = F.shape
    y_end = int(min(H - 1, y_sh + 1.1 * span))
    min_run = max(3, int(round(0.025 * span)))       # 3 px on golden (span 118), as in the earlier audit
    sep = {"small-x": [], "large-x": []}
    torso_rows = {}
    for y in range(int(y_sh), y_end + 1):
        runs = runs_of(F[y])
        if not runs:
            continue
        inside = [r for r in runs if r[0] <= x_c <= r[1]]
        torso = inside[0] if inside else min(runs, key=lambda r: abs((r[0] + r[1]) / 2 - x_c))
        left = [r for r in runs if r[1] < torso[0] - 1 and (r[1] - r[0] + 1) >= min_run]
        right = [r for r in runs if r[0] > torso[1] + 1 and (r[1] - r[0] + 1) >= min_run]
        torso_rows[y] = (torso, bool(left), bool(right))
        sep["small-x"].append((y, bool(left), ((max(r[1] for r in left) + torso[0]) / 2.0) if left else None))
        sep["large-x"].append((y, bool(right), ((torso[1] + min(r[0] for r in right)) / 2.0) if right else None))
    apex, persist = {}, {}
    for side, lst in sep.items():
        apex[side], persist[side] = None, 0
        for k in range(len(lst) - 2):
            if lst[k][1] and lst[k + 1][1] and lst[k + 2][1]:
                apex[side] = (float(lst[k][2]), float(lst[k][0]))
                persist[side] = sum(1 for e in lst[k:] if e[1])      # rows with a visible gap from the apex down
                break
    axis = None
    if apex["small-x"] and apex["large-x"]:
        y0 = int(max(apex["small-x"][1], apex["large-x"][1])) + int(round(0.07 * span))
        nrows = int(round(0.85 * span))
        centres = [(y, (torso_rows[y][0][0] + torso_rows[y][0][1]) / 2.0) for y in range(y0, y0 + nrows)
                   if y in torso_rows and torso_rows[y][1] and torso_rows[y][2]]
        axis = fit_axis(centres)
    depth = {k: ((v[1] - y_sh) / span if v else None) for k, v in apex.items()}
    plausible = bool(all(v is not None and DEPTH_MIN <= v <= DEPTH_MAX for v in depth.values()))
    return dict(apex=apex, persist=persist, axis=axis, min_run=min_run, scan=(int(y_sh), y_end), depth=depth,
                plausible=plausible)


# ----------------------------------------------------------------------------- drawing
def _x(d, p, r, color, w=2):
    d.line([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=color, width=w)
    d.line([p[0] - r, p[1] + r, p[0] + r, p[1] - r], fill=color, width=w)


def _o(d, p, r, color, w=2):
    d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], outline=color, width=w)


def draw_armpit_zoom(img, F, F6, R, label):
    """Unmarked 3x crops of the armpit rows: raw photo | foreground sum>36 | foreground sum>6 (when available).
    The rows are labelled with ticks in the margin only, so the pixels themselves are never covered."""
    H, W = F.shape
    ys = [a[1] for a in R["apex"]["apex"].values() if a]
    lo6 = R.get("apex_lowthr")
    if lo6:
        ys += [a[1] for a in lo6["apex"].values() if a]
    ys += [d["y"] for _, d in R["detector_marks"]]
    if not ys:
        return None
    y0, y1 = int(max(0, min(ys) - 30)), int(min(H, max(ys) + 45))
    k = 3
    rgb = img.convert("RGB").crop((0, y0, W, y1)).resize((W * k, (y1 - y0) * k), Image.NEAREST)
    panels = [rgb, Image.fromarray((F[y0:y1].astype(np.uint8) * 255)).convert("RGB").resize((W * k, (y1 - y0) * k), Image.NEAREST)]
    if F6 is not None:
        panels.append(Image.fromarray((F6[y0:y1].astype(np.uint8) * 255)).convert("RGB").resize((W * k, (y1 - y0) * k), Image.NEAREST))
    sheet = Image.new("RGB", (sum(p.size[0] for p in panels) + 8 * (len(panels) - 1) + 40, panels[0].size[1] + 22), (30, 30, 30))
    d = ImageDraw.Draw(sheet)
    d.text((4, 4), f"{label} rows {y0}..{y1}   photo | foreground sum>36 | foreground sum>6      ticks at the left edge: magenta = apex sum>36, green = apex sum>6, cyan = detector row", fill=(255, 255, 255))
    xo = 40
    for p in panels:
        sheet.paste(p, (xo, 22)); xo += p.size[0] + 8
    def tick(y, col):
        yy = 22 + (y - y0) * k
        d.line([0, yy, 36, yy], fill=col, width=2)
    for a in R["apex"]["apex"].values():
        if a: tick(a[1], (255, 60, 255))
    if lo6:
        for a in lo6["apex"].values():
            if a: tick(a[1], (60, 255, 60))
    for _, dd in R["detector_marks"]:
        tick(dd["y"], (0, 255, 255))
    return sheet


def draw_panel(img, F, torso, arms, R, label):
    H, W = F.shape
    ys, xs = np.where(F)
    pad = 10
    x0, x1 = max(0, int(xs.min()) - pad), min(W, int(xs.max()) + pad)
    y0, y1 = max(0, int(ys.min()) - pad), min(H, int(ys.max()) + pad)
    k = max(1, int(round(420.0 / max(1, (y1 - y0)))))
    k = min(k, 4)
    S = lambda p: ((p[0] - x0) * k, (p[1] - y0) * k)
    rgba = img.convert("RGBA")
    base = Image.new("RGB", rgba.size, (128, 128, 128))
    alpha = rgba.split()[3]
    base.paste(rgba.convert("RGB"), (0, 0), alpha if alpha.getextrema()[0] < 128 else None)
    panels = []
    for kind in ("photo", "fg", "masks"):
        if kind == "photo":
            im = base
        elif kind == "fg":
            im = Image.fromarray((F.astype(np.uint8) * 255)).convert("RGB")
        else:
            t = np.zeros((H, W, 3), np.uint8)
            t[F] = (60, 60, 60)
            t[torso] = (230, 60, 60)
            t[arms] = (60, 120, 255)
            im = Image.fromarray(t)
        im = im.crop((x0, y0, x1, y1)).resize(((x1 - x0) * k, (y1 - y0) * k), Image.NEAREST if kind != "photo" else Image.LANCZOS)
        d = ImageDraw.Draw(im)
        if kind != "fg":
            for side, col in (("small-x", (255, 60, 255)), ("large-x", (255, 60, 255))):
                if R["apex"]["apex"][side]:
                    _x(d, S(R["apex"]["apex"][side]), 9, col, 3)
            lo6 = R.get("apex_lowthr")
            if lo6:
                for side in ("small-x", "large-x"):
                    if lo6["apex"][side]:
                        _x(d, S(lo6["apex"][side]), 7, (60, 255, 60), 2)
        else:                                               # keep the silhouette clean: only edge ticks at the apex rows
            for side in ("small-x", "large-x"):
                if R["apex"]["apex"][side]:
                    yy = S(R["apex"]["apex"][side])[1]
                    d.line([0, yy, 8, yy], fill=(255, 60, 255), width=2)
            lo6 = R.get("apex_lowthr")
            if lo6:
                for side in ("small-x", "large-x"):
                    if lo6["apex"][side]:
                        yy = S(lo6["apex"][side])[1]
                        d.line([im.size[0] - 9, yy, im.size[0] - 1, yy], fill=(60, 255, 60), width=2)
        for side_, dd in (R["detector_marks"] if kind != "fg" else []):
            if dd["x"] is not None:
                _o(d, S((dd["x"], dd["y"])), 9, (0, 255, 255), 3)
            else:                                           # row only: a cyan dash on that side of the body
                xm = R["x_c"] + (-0.45 if side_ == "left" else 0.45) * R["span"]
                d.line([S((xm - 0.12 * R["span"], dd["y"])), S((xm + 0.12 * R["span"], dd["y"]))], fill=(0, 255, 255), width=3)
        for p in R["pipeline_underarms"]:
            if p is not None:
                _o(d, S(p), 13, (255, 220, 0), 2)
        if kind == "masks" and R["masks"]["uncovered_bbox"]:
            bx = R["masks"]["uncovered_bbox"]
            d.rectangle([S((bx[0], bx[1])), S((bx[2], bx[3]))], outline=(255, 255, 0), width=2)
        d.text((4, 4), f"{kind}", fill=(255, 255, 255))
        panels.append(im)
    Wt = sum(p.size[0] for p in panels) + 8 * (len(panels) - 1)
    Ht = max(p.size[1] for p in panels) + 22
    sheet = Image.new("RGB", (Wt, Ht), (30, 30, 30))
    xo = 0
    for p in panels:
        sheet.paste(p, (xo, 22)); xo += p.size[0] + 8
    ImageDraw.Draw(sheet).text((4, 4), f"{label}   magenta X = silhouette apex   cyan o = detector result   yellow o = pipeline property   "
                                       f"red = torso mask  blue = upper-arm mask  yellow box = central foreground NOT covered by masks", fill=(255, 255, 255))
    return sheet


# ----------------------------------------------------------------------------- one avatar
def audit_avatar(user_id, label):
    bar = "=" * 70
    print(f"\n{bar}\nAVATAR  {label}   (user id: {user_id or 'default / golden'})\n{bar}")
    try:
        avatar = build_avatar_representation(avatar_user_id=user_id) if user_id else build_avatar_representation()
    except Exception as exc:                                                   # noqa: BLE001
        print(f"NOT TESTABLE: avatar representation failed: {type(exc).__name__}: {exc}")
        return None

    R = dict(label=label, user_id=user_id)
    for name in ("avatar_id", "version", "avatar_version", "id"):
        v, e = guarded(lambda n=name: getattr(avatar, n))
        if e is None and v is not None and not callable(v):
            print(f"  {name} = {v}")
            R[name] = str(v)

    img = avatar.image
    F, how, arr = foreground(img)
    H, W = F.shape
    torso = np.asarray(avatar.torso_mask) > 0
    arms = np.asarray(avatar.upper_arms_mask) > 0
    if not F.any():
        alt_n = int((arr[:, :, :3].astype(int).sum(axis=2) > 6).sum())
        print(f"NOT TESTABLE: the foreground is empty (source: {how}). With the threshold lowered to sum>6 there would be {alt_n} foreground pixels"
              + (" - the image is almost entirely near-black: dark skin on a black background is very likely what is happening." if alt_n > 0 and "NO usable alpha" in how else "."))
        return None

    # --- shoulders (guarded); fall back to the silhouette only with an explicit label
    sh_raw, sh_err = guarded(lambda: (avatar.left_shoulder, avatar.right_shoulder))
    ys_f, xs_f = np.where(F)
    if sh_raw is not None:
        a, b = sorted([tuple(map(float, sh_raw[0][:2])), tuple(map(float, sh_raw[1][:2]))])
        sh = (a, b)
        x_c = (a[0] + b[0]) / 2.0
        y_sh = (a[1] + b[1]) / 2.0
        span = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        sh_note = "pipeline shoulders"
    else:
        sh = None
        x_c = float(np.median(xs_f))
        y_sh = float(ys_f.min() + 0.2 * (ys_f.max() - ys_f.min()))
        span = float(0.5 * (xs_f.max() - xs_f.min()))
        sh_note = f"ESTIMATED from the silhouette (shoulders unavailable: {sh_err})"
    R.update(shoulders=sh, x_c=x_c, y_sh=y_sh, span=span)
    print(f"  image {W}x{H}; foreground source: {how}; {sh_note}")
    print(f"  shoulder span {span:.0f}px -> windows: scan {int(y_sh)}..{int(min(H - 1, y_sh + 1.1 * span))}, centre x {x_c:.1f}")

    # --- foreground & masks
    fd = foreground_diagnostics(F, how, arr, x_c, y_sh, span)
    md = mask_diagnostics(F, torso, arms, x_c, span, fd["band"])
    R["foreground"], R["masks"] = fd, md
    print(f"\n  FOREGROUND   area {fd['area']} px ({100 * fd['area_frac']:.1f}% of image); components {fd['components']} (largest {100 * fd['largest_frac']:.1f}%, torso-band share of the main one {100 * fd['band_main_frac']:.1f}%); "
          f"enclosed holes {100 * fd['holes_frac']:.2f}% of the figure / {100 * fd['holes_central_frac']:.2f}% of the central torso strip; centreline missing {100 * fd['centreline_missing']:.1f}% of torso rows")
    print(f"               mean luminance {fd['mean_lum']:.0f}/255; near-black pixels {100 * fd['near_black_frac']:.1f}%; contrast vs black {fd['contrast_vs_black']:.0f}, vs white {fd['contrast_vs_white']:.0f}; "
          f"convex-hull gap {100 * fd['hull_gap_frac']:.1f}%")
    if "NO usable alpha" in how:
        r_ = fd["alt_area_ratio"]
        print(f"               NOTE: no alpha channel. Lowering the non-black threshold (sum>36 -> sum>6) changes the foreground area by x{r_:.2f}"
              + ("  -> THRESHOLD-SENSITIVE: dark pixels are being lost (dark skin on black is the likely cause)" if r_ > 1.3 else "  -> stable (dark pixels are not being lost)"))
    print(f"  MASKS        torso {md['torso_area']} px ({100 * md['torso_frac_of_fg']:.1f}% of fg), upper arms {md['arms_area']} px ({100 * md['arms_frac_of_fg']:.1f}% of fg)")
    print(f"               central foreground in the torso band not covered by either mask: {100 * md['uncovered_frac']:.1f}%"
          + (f"   bbox x{md['uncovered_bbox'][0]}-{md['uncovered_bbox'][2]}, y{md['uncovered_bbox'][1]}-{md['uncovered_bbox'][3]}" if md["uncovered_bbox"] else ""))
    if not fd["fg_intact"]:
        verdict = "FOREGROUND FAILURE: the silhouette itself is damaged - the apex estimate below is NOT trustworthy"
        if fd["alt_area_ratio"] is not None and fd["alt_area_ratio"] > 1.3:
            verdict += " (consistent with dark skin on a black background: the foreground threshold is losing it)"
    elif md["mask_hole"]:
        verdict = "MASK FAILURE (SCHP): foreground intact, masks leave a hole - the pipeline detector fails, the silhouette apex is valid"
    else:
        verdict = "masks and foreground both fine"
    R["verdict"] = verdict
    print(f"  V verdict    {verdict}")

    # --- pipeline detector, called directly, None tolerated
    det = {}
    R["detector_raw"] = {}
    for side in ("left", "right"):
        v, e = guarded(lambda s=side: find_semantic_underarm(torso, arms, W, s))
        R["detector_raw"][side] = short(v) if e is None else "RAISED " + e
        if e is not None:
            det[side] = None
            print(f"  detector find_semantic_underarm(side={side!r}): RAISED [{e}]")
            continue
        print(f"  detector find_semantic_underarm(side={side!r}) raw return: {short(v)}   (type {type(v).__name__})")
        try:
            det[side] = parse_detector(v)
        except ValueError as ve:
            det[side] = None
            R.setdefault("detector_shape_problem", []).append(side)
            print(f"     -> {ve}: treated as 'no usable point' for the comparison below; send me the raw line above")
            continue
        d_ = det[side]
        if d_ is None:
            print("     -> None (the detector found no underarm)")
        elif d_["x"] is None:
            rr = d_["rows"]
            print(f"     -> underarm ROW {d_['y']:.0f} (the detector gives a row, not an x)"
                  + (f"; its adjacent-row list spans {rr[0]}..{rr[1]} ({rr[2]} rows)" if rr else ""))
        else:
            print(f"     -> point ({d_['x']:.1f}, {d_['y']:.1f})")
    R["detector"] = det
    R["detector_marks"] = [(side, det[side]) for side in ("left", "right") if det[side] is not None]

    # --- the pipeline PROPERTIES (guarded: these are what raise 422 UNDERARM_NOT_DETECTED)
    pl, pr = guarded(lambda: avatar.left_underarm)[0], guarded(lambda: avatar.right_underarm)[0]
    pe = guarded(lambda: avatar.left_underarm)[1] or guarded(lambda: avatar.right_underarm)[1]
    pu = []
    for raw_ in (pl, pr):
        try:
            pu.append(pt(raw_))
        except ValueError as ve:
            pe = pe or str(ve)
            pu.append(None)
            print(f"  pipeline property raw value: {short(raw_)} ({ve})")
    R["pipeline_underarms"] = pu
    print("  pipeline properties left/right_underarm: " + ("available" if pe is None else f"RAISE ({pe})"))

    # --- silhouette apex
    ap = silhouette_apex(F, x_c, y_sh, span)
    R["apex"] = ap
    for side in ("small-x", "large-x"):
        p = ap["apex"][side]
        print(f"  silhouette apex {side}: " + (f"({p[0]:.1f}, {p[1]:.0f})  gap visible for {ap['persist'][side]} rows below it" if p else "NOT FOUND (the arm never separates from the torso in the scan window)"))
    both = bool(ap["apex"]["small-x"] and ap["apex"]["large-x"])
    print(f"  A1 apex on both sides: {'YES' if both else 'NO'}")
    if not both:
        print("     (the arms stay in contact with the torso through the whole scan window, or the visible arm is narrower than "
              f"{ap['min_run']}px: this pose does not give the visible arm/torso gap the apex method needs - arms must be held away from the body)")
    if both:
        dsx, dlx = ap["depth"]["small-x"], ap["depth"]["large-x"]
        print(f"  A4 apex depth below the shoulder line: {dsx:.2f} / {dlx:.2f} of the shoulder span (plausible {DEPTH_MIN:.2f}..{DEPTH_MAX:.2f}) -> "
              + ("PLAUSIBLE" if ap["plausible"] else "IMPLAUSIBLE - probably not an armpit"))
    for side_ in ("left", "right"):
        if det[side_] is not None:
            print(f"     (for comparison, detector {side_} row {det[side_]['y']:.0f} is at depth {(det[side_]['y'] - y_sh) / span:.2f})")
    if both:
        a_deg = ang(ap["apex"]["small-x"], ap["apex"]["large-x"])
        R["apex_line_deg"] = a_deg
        print(f"  A2 apex line: vertical diff {ap['apex']['large-x'][1] - ap['apex']['small-x'][1]:+.0f}px, angle {a_deg:+.2f} deg -> {'LEVEL' if abs(a_deg) <= TILT_TOL else 'TILTED'} (tol {TILT_TOL} deg)")
    if ap["axis"]:
        print(f"     torso centreline lean {ap['axis'][0]:+.2f} deg ({ap['axis'][1]} rows, residual std {ap['axis'][2]:.2f}px)")

    # --- A3: detector (a row) and pipeline properties (x, y) vs the silhouette apex
    det_by_side = {"small-x": det["left"], "large-x": det["right"]}
    pipe_by_side = {}
    for q in R["pipeline_underarms"]:
        if q is not None:
            pipe_by_side["small-x" if q[0] < x_c else "large-x"] = dict(x=q[0], y=q[1])
    for src, table in (("detector", det_by_side), ("pipeline", pipe_by_side)):
        if src == "pipeline" and pe is not None:
            continue
        for side in ("small-x", "large-x"):
            a_ = ap["apex"][side]
            d_ = table.get(side)
            if a_ is None:
                msg = "no apex to compare against"
            elif d_ is None:
                msg = "DETECTOR FAILED (None) - no landmark to compare" if src == "detector" else "no landmark"
            else:
                dy = d_["y"] - a_[1]
                where = f"x {d_['x']:.1f}, " if d_["x"] is not None else ""
                msg = (f"{where}y {d_['y']:.0f} vs apex ({a_[0]:.1f}, {a_[1]:.0f}): dy {dy:+.1f}px"
                       + (f" dx {d_['x'] - a_[0]:+.1f}px" if d_["x"] is not None else "  (row only)")
                       + " -> " + ("AGREES" if abs(dy) <= AGREE_PX else "DISAGREES"))
            R.setdefault("A3", {})[f"{src}:{side}"] = msg
            print(f"  A3 {src:<8} {side}: {msg}")

    # --- A5: apex stability under a lower foreground threshold (a dark crease can read as an arm/torso gap)
    ap6, stable = None, True
    R["A5"] = None
    if "NO usable alpha" in how:
        F6 = arr[:, :, :3].astype(int).sum(axis=2) > 6
        n6, lab6, st6, _ = cv2.connectedComponentsWithStats(F6.astype(np.uint8), connectivity=8)
        if n6 > 1:
            F6 = lab6 == (1 + int(np.argmax(st6[1:, cv2.CC_STAT_AREA])))
        ap6 = silhouette_apex(F6, x_c, y_sh, span)
        R["apex_lowthr"] = ap6
        lost = F6 & ~F
        y0b, y1b = fd["band"]
        lost_band = int(lost[y0b:y1b + 1].sum())
        parts, moved = [], []
        for side in ("small-x", "large-x"):
            a36, a6 = ap["apex"][side], ap6["apex"][side]
            if a36 is None or a6 is None:
                parts.append(f"{side}: sum>36 {'none' if a36 is None else int(a36[1])} / sum>6 {'none' if a6 is None else int(a6[1])}")
                moved.append(None)
            else:
                moved.append(a6[1] - a36[1])
                parts.append(f"{side}: y {int(a36[1])} -> {int(a6[1])} ({a6[1] - a36[1]:+.0f}px, x {a36[0]:.1f} -> {a6[0]:.1f})")
        stable = bool(all(m is not None and abs(m) <= AGREE_PX for m in moved))
        R["A5"] = dict(moved=moved, stable=stable, lost_band_px=lost_band)
        print(f"  A5 apex under the lower threshold (sum>6, {lost_band} extra foreground px in the torso band): " + "; ".join(parts))
        print("     -> " + ("STABLE: the apex does not depend on dark pixels" if stable else
              "UNSTABLE: the sum>36 apex is probably a DARK CREASE (dark skin between arm and chest read as background), not the armpit; "
              "the sum>6 apex below is the one to trust" ))
        if not stable:
            for side in ("small-x", "large-x"):
                a6 = ap6["apex"][side]
                if a6:
                    print(f"     low-threshold apex {side}: ({a6[0]:.1f}, {a6[1]:.0f}), depth {(a6[1] - y_sh) / span:.2f}; gap visible for {ap6['persist'][side]} rows below it")
            for side_, key in (("left", "small-x"), ("right", "large-x")):
                a6 = ap6["apex"][key]
                d_ = det[side_]
                if a6 and d_ is not None:
                    print(f"     detector {side_} row {d_['y']:.0f} vs low-threshold apex {a6[1]:.0f}: dy {d_['y'] - a6[1]:+.1f}px -> {'AGREES' if abs(d_['y'] - a6[1]) <= AGREE_PX else 'DISAGREES'}")
                elif a6 and d_ is None:
                    print(f"     detector {side_}: None, low-threshold apex {a6[1]:.0f}")
            if ap6["apex"]["small-x"] and ap6["apex"]["large-x"]:
                R["apex_line_deg_lowthr"] = ang(ap6["apex"]["small-x"], ap6["apex"]["large-x"])
                print(f"     low-threshold apex line: angle {R['apex_line_deg_lowthr']:+.2f} deg, plausible depth: {ap6['plausible']}")

    # --- golden drift guard (Y only; the recorded C2 numbers kept the pipeline X)
    if user_id is None:
        if both:
            d_s = abs(ap["apex"]["small-x"][1] - GOLDEN_APEX_Y[0])
            d_l = abs(ap["apex"]["large-x"][1] - GOLDEN_APEX_Y[1])
            ok = bool(d_s <= GOLDEN_APEX_TOL and d_l <= GOLDEN_APEX_TOL)
            R["drift_ok"] = ok
            print(f"  D  golden drift guard (Y): apex Y moved {d_s:.1f}px / {d_l:.1f}px from the recorded {GOLDEN_APEX_Y[0]:.0f} / {GOLDEN_APEX_Y[1]:.0f} -> "
                  + ("OK" if ok else "DRIFT - the scaled windows or the avatar changed; STOP and report before using this script"))
            for side in ("small-x", "large-x"):
                if side in pipe_by_side and pe is None:
                    print(f"     (information: apex X {ap['apex'][side][0]:.1f} vs pipeline X {pipe_by_side[side]['x']:.1f}, dx {ap['apex'][side][0] - pipe_by_side[side]['x']:+.1f}px; "
                          "C2 used the pipeline X)")
        else:
            R["drift_ok"] = False
            print("  D  golden drift guard: apex NOT FOUND on the golden avatar -> DRIFT; STOP and report")

    # --- usability for the C2 experiment
    usable = bool(both and ap["plausible"] and fd["fg_intact"] and stable)
    if not stable and ap6 is not None and ap6["plausible"] and all(ap6["apex"].values()) and fd["fg_intact"]:
        R["recommended_apex"] = "low-threshold (sum>6)"
        R["c2_usable_with_lowthr_apex"] = True
    R["c2_usable"] = usable
    if usable:
        print("  => the silhouette apex is usable on this avatar" + ("; C2 as written needs the pipeline underarm X, which this avatar cannot provide (its underarm properties raise) - the next script would take the apex X instead" if pe is not None else ""))
    elif R.get("c2_usable_with_lowthr_apex"):
        print("  => the sum>36 apex is NOT usable on this avatar (dark crease), but the LOW-THRESHOLD (sum>6) apex above is plausible, "
              "the foreground is intact and the detector's row agrees or can be compared with it: use the sum>6 apex"
              + ("; C2 as written also needs the pipeline underarm X, which this avatar cannot provide - the next script would take the apex X" if pe is not None else ""))
    else:
        print("  => the C2 check is NOT possible on this avatar (" + ("foreground damaged" if not fd["fg_intact"] else ("no apex on both sides" if not both else ("apex depth implausible" if not ap["plausible"] else "sum>36 apex is unstable (dark crease) - use the low-threshold apex"))) + ")")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    png = f"{OUTPUT_DIR}/avatar_audit_{safe}.png"
    draw_panel(img, F, torso, arms, R, label).save(png)
    print(f"  Saved: {png}")
    F6_for_zoom = None
    if R.get("apex_lowthr") is not None:
        F6_for_zoom = arr[:, :, :3].astype(int).sum(axis=2) > 6
    zoom = draw_armpit_zoom(img, F, F6_for_zoom, R, label)
    if zoom is not None:
        zpath = f"{OUTPUT_DIR}/avatar_audit_{safe}_armpit.png"
        zoom.save(zpath)
        print(f"  Saved: {zpath}  (clean 3x crop of the armpit rows - look at this one with your own eyes)")

    def jsonable(o):
        if isinstance(o, dict):
            return {str(k): jsonable(v) for k, v in o.items() if k not in ("F",)}
        if isinstance(o, (list, tuple)):
            return [jsonable(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    with open(f"{OUTPUT_DIR}/avatar_audit_{safe}.json", "w") as fh:
        json.dump(jsonable(R), fh, indent=1)
    print(f"  Saved: {OUTPUT_DIR}/avatar_audit_{safe}.json  (paste this back; it is small)")
    return R


def main(ids):
    print(__doc__)
    results = []
    for tok in ids:
        uid = None if tok in ("golden", "default") else tok
        label = "golden" if uid is None else uid[:8]
        r = audit_avatar(uid, label)
        if r:
            results.append(r)
    if len(results) > 1:
        bar = "=" * 70
        print(f"\n{bar}\nCROSS-AVATAR SUMMARY\n{bar}")
        print(f"  {'avatar':<12}{'size':<12}{'apex L/R':<26}{'apex deg':>9}{'mask hole':>11}{'fg intact':>11}   verdict")
        for r in results:
            a = r["apex"]["apex"]
            fa = lambda p: "({:.0f},{:.0f})".format(p[0], p[1]) if p else "none"
            adeg = r.get("apex_line_deg")
            adeg_s = "{:+.1f}".format(adeg) if adeg is not None else "n/a"
            both_s = fa(a["small-x"]) + " " + fa(a["large-x"])
            print("  {:<12}{:<12}{:<26}{:>9}{:>11}{:>11}   {}".format(
                r["label"], str(r["foreground"]["size"]), both_s, adeg_s,
                "YES" if r["masks"]["mask_hole"] else "no",
                "yes" if r["foreground"]["fg_intact"] else "NO",
                r["verdict"].split(":")[0]))


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser(description="Read-only audit of one or more avatars (silhouette apex vs pipeline underarm).")
    ap_.add_argument("--avatar-user-id", nargs="*", default=["golden"], help="avatar user ids; 'golden' = the default avatar")
    a_ = ap_.parse_args()
    main(a_.avatar_user_id) 