"""
Try-On Studio B10b (2026-11-26): body OUTLINE points, measured from the avatar's own
cut-out (the transparent background of the stage photo).

MediaPipe's points are joint centres (the skeleton). Clothes are placed by the body's
edges, so these are measured from the photo's real pixels: walk from a joint outward
until the body ends. Nothing is estimated or invented - a point that can't be measured
is left out.

Points (fractions 0-1 of the stage photo, top-left origin; _l/_r = the PERSON's left/right):
    neck, neck_edge_l/r        where the neck meets the shoulders (centre + sides)
    shoulder_edge_l/r          outside of the shoulder, at shoulder-joint height
    shoulder_top_l/r           top of the shoulder, straight above the joint
    waist_edge_l/r             sides of the narrowest part of the torso above the hips
    hip_edge_l/r               sides of the body at hip-joint height
"""

import io

import numpy as np
from PIL import Image

from shared.logger import get_logger

logger = get_logger(__name__)

ALPHA_MIN = 128      # alpha >= this counts as body
GAP = 3              # this many see-through pixels in a row = the body has ended
NECK_SLACK = 1.3     # neck base = lowest row still <= 1.3 x the narrowest neck row
MIN_VISIBILITY = 0.5


def _edge(mask, x, y, step):
    h, w = mask.shape
    if not (0 <= x < w and 0 <= y < h) or not mask[y, x]:
        return None
    last, gap = x, 0
    while 0 <= x < w:
        if mask[y, x]:
            last, gap = x, 0
        else:
            gap += 1
            if gap >= GAP:
                break
        x += step
    return last


def _top(mask, x, y):
    h, w = mask.shape
    if not (0 <= x < w and 0 <= y < h) or not mask[y, x]:
        return None
    last, gap = y, 0
    while y >= 0:
        if mask[y, x]:
            last, gap = y, 0
        else:
            gap += 1
            if gap >= GAP:
                break
        y -= 1
    return last


def _row(mask, cx, y):
    l, r = _edge(mask, cx, y, -1), _edge(mask, cx, y, 1)
    return (l, r) if l is not None and r is not None else None


def measure_outline(stage_bytes: bytes, pose_keypoints: dict | None) -> dict | None:
    """Outline points for the stage photo, or None if it can't be measured.
    Never raises - outline points are an enhancement, never a blocker."""
    if not pose_keypoints:
        return None
    try:
        img = Image.open(io.BytesIO(stage_bytes)).convert("RGBA")
        mask = np.asarray(img)[:, :, 3] >= ALPHA_MIN
        h, w = mask.shape
        if mask.all():
            logger.warning("Outline - the stage photo has no transparent background")
            return None

        def px(name):
            p = pose_keypoints.get(name)
            if not p or float(p.get("visibility", 1)) < MIN_VISIBILITY:
                return None
            return (min(w - 1, max(0, int(float(p["x"]) * w))),
                    min(h - 1, max(0, int(float(p["y"]) * h))))

        def frac(x, y):
            return [round(x / w, 4), round(y / h, 4)]

        out = {}
        sl, sr = px("left_shoulder"), px("right_shoulder")
        hl, hr = px("left_hip"), px("right_hip")
        if not (sl and sr):
            return None
        out_l = 1 if sl[0] > sr[0] else -1      # image direction of the person's left

        # shoulders
        for side, j, d in (("l", sl, out_l), ("r", sr, -out_l)):
            ex = _edge(mask, j[0], j[1], d)
            ty = _top(mask, j[0], j[1])
            if ex is not None:
                out[f"shoulder_edge_{side}"] = frac(ex, j[1])
            if ty is not None:
                out[f"shoulder_top_{side}"] = frac(j[0], ty)

        # neck base
        cx = (sl[0] + sr[0]) // 2
        sy = min(sl[1], sr[1])
        mouth = [p for p in (px("mouth_left"), px("mouth_right")) if p]
        nose = px("nose")
        if mouth:
            y0 = sum(p[1] for p in mouth) // len(mouth)
        elif nose:
            y0 = nose[1]
        else:
            y0 = None
        if y0 is not None:
            rows = []
            for y in range(y0 + 1, sy):
                r = _row(mask, cx, y)
                if r:
                    rows.append((r[1] - r[0], y, r[0], r[1]))
            if rows:
                narrow = min(rows)[0]
                for width, y, l, r in reversed(rows):      # from the shoulders upward
                    if width <= NECK_SLACK * narrow:
                        out["neck"] = frac((l + r) / 2, y)
                        out["neck_edge_l" if out_l == 1 else "neck_edge_r"] = frac(r, y)
                        out["neck_edge_r" if out_l == 1 else "neck_edge_l"] = frac(l, y)
                        break

        if hl and hr:
            # hips
            for side, j, d in (("l", hl, out_l), ("r", hr, -out_l)):
                ex = _edge(mask, j[0], j[1], d)
                if ex is not None:
                    out[f"hip_edge_{side}"] = frac(ex, j[1])
            # waist: narrowest torso row in the lower 60 % between shoulders and hips
            hy = (hl[1] + hr[1]) // 2
            hcx = (hl[0] + hr[0]) // 2
            trows = []
            for y in range(sy + int(0.4 * (hy - sy)), hy):
                r = _row(mask, hcx, y)
                if r:
                    trows.append((r[1] - r[0], y, r[0], r[1]))
            if trows:
                _, y, l, r = min(trows)
                out["waist_edge_l" if out_l == 1 else "waist_edge_r"] = frac(r, y)
                out["waist_edge_r" if out_l == 1 else "waist_edge_l"] = frac(l, y)

        logger.info("Outline - %d points measured", len(out))
        return out or None

    except Exception:
        logger.exception("Outline measurement failed")
        return None