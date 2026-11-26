"""
B10b prototype (Try-On Studio, 2026-11-26, costs $0, changes nothing).

MediaPipe's points are JOINT CENTRES (the skeleton). A garment is placed by the body's
EDGES: where the neck meets the shoulders, the outside of the shoulders, the sides of
the hips. This measures those edges from the avatar's own cut-out (the transparent
background of processed_photo_url) - real pixels of the photo, no guessing - and
draws them next to the joints.

    python -m prototypes.t2_2.b10b_outline
    python -m prototypes.t2_2.b10b_outline --user <user_id>

Writes prototypes/t2_2/out/b10b/<first 8 of user id>_outline.png and .json
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

KEJI = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
OUT = Path(__file__).resolve().parent / "out" / "b10b"

ALPHA_MIN = 128      # a pixel with alpha >= this is "body"
GAP = 3              # this many see-through pixels in a row = the body has ended
NECK_SLACK = 1.3     # neck base = lowest row whose width is still <= 1.3 x the narrowest neck row
BACKDROP = (232, 232, 234)


def body_mask(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGBA"))[:, :, 3] >= ALPHA_MIN


def _edge(mask, x, y, step):
    """From (x, y) inside the body, walk sideways (step -1 / +1) to the last body pixel."""
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
    """From (x, y) inside the body, walk up to the top body pixel."""
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


def measure(img: Image.Image, kp: dict) -> dict:
    m = body_mask(img)
    h, w = m.shape

    def px(name):
        p = kp.get(name)
        if not p or float(p.get("visibility", 1)) < 0.5:
            return None
        return (min(w - 1, max(0, int(float(p["x"]) * w))),
                min(h - 1, max(0, int(float(p["y"]) * h))))

    def frac(x, y):
        return [round(x / w, 4), round(y / h, 4)]

    out, notes = {}, []
    sl, sr = px("left_shoulder"), px("right_shoulder")
    hl, hr = px("left_hip"), px("right_hip")
    if not (sl and sr):
        return {"points": {}, "notes": ["shoulders not visible"]}
    out_l = 1 if sl[0] > sr[0] else -1          # direction of the person's left in the image

    # --- shoulders: outside edge at joint height, and top of the shoulder above the joint
    for side, j, d in (("l", sl, out_l), ("r", sr, -out_l)):
        ex = _edge(m, j[0], j[1], d)
        ty = _top(m, j[0], j[1])
        if ex is not None:
            out[f"shoulder_edge_{side}"] = frac(ex, j[1])
        if ty is not None:
            out[f"shoulder_top_{side}"] = frac(j[0], ty)

    # --- neck: narrowest row between mouth and shoulders; neck base = where it starts to widen
    cx = (sl[0] + sr[0]) // 2
    sy = min(sl[1], sr[1])
    mouth = [p for p in (px("mouth_left"), px("mouth_right")) if p]
    nose = px("nose")
    y0 = (sum(p[1] for p in mouth) // len(mouth)) if mouth else (nose[1] if nose else sy - (sy // 4))
    rows = []
    for y in range(y0 + 1, sy):
        r = _row(m, cx, y)
        if r:
            rows.append((r[1] - r[0], y, r[0], r[1]))
    if rows:
        narrow = min(rows)
        base = None
        for width, y, l, r in reversed(rows):            # from the shoulders upward
            if width <= NECK_SLACK * narrow[0]:
                base = (width, y, l, r)
                break
        if base:
            out["neck"] = frac((base[2] + base[3]) / 2, base[1])
            out["neck_edge_l" if out_l == 1 else "neck_edge_r"] = frac(base[3], base[1])
            out["neck_edge_r" if out_l == 1 else "neck_edge_l"] = frac(base[2], base[1])
    else:
        notes.append("neck not measurable")

    # --- hips: sides of the body at hip-joint height (first gap = torso edge, not the arm)
    if hl and hr:
        for side, j, d in (("l", hl, out_l), ("r", hr, -out_l)):
            ex = _edge(m, j[0], j[1], d)
            if ex is not None:
                out[f"hip_edge_{side}"] = frac(ex, j[1])

        # --- waist: narrowest torso row in the lower 60 % between shoulders and hips
        hy = (hl[1] + hr[1]) // 2
        hcx = (hl[0] + hr[0]) // 2
        trows = []
        for y in range(sy + int(0.4 * (hy - sy)), hy):
            r = _row(m, hcx, y)
            if r:
                trows.append((r[1] - r[0], y, r[0], r[1]))
        if trows:
            wst = min(trows)
            out["waist_edge_l" if out_l == 1 else "waist_edge_r"] = frac(wst[3], wst[1])
            out["waist_edge_r" if out_l == 1 else "waist_edge_l"] = frac(wst[2], wst[1])

    # --- how far apart joints and edges are (fractions of the image width)
    def span(a, b):
        return round(abs(out[a][0] - out[b][0]), 4) if a in out and b in out else None

    compare = {
        "shoulder_joints_w": round(abs(sl[0] - sr[0]) / w, 4),
        "shoulder_edges_w": span("shoulder_edge_l", "shoulder_edge_r"),
        "hip_joints_w": round(abs(hl[0] - hr[0]) / w, 4) if hl and hr else None,
        "hip_edges_w": span("hip_edge_l", "hip_edge_r"),
        "old_neck_y": round(sy / h, 4),
        "new_neck_y": out.get("neck", [None, None])[1],
    }
    return {"image_w": w, "image_h": h, "points": out, "compare": compare, "notes": notes}


def draw(img: Image.Image, kp: dict, res: dict) -> Image.Image:
    rgba = img.convert("RGBA")
    flat = Image.new("RGB", rgba.size, BACKDROP)
    flat.paste(rgba, mask=rgba.split()[3])
    w, h = flat.size
    d = ImageDraw.Draw(flat)
    r = max(3, w // 120)
    for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip", "nose"):
        p = kp.get(name)
        if p:
            x, y = float(p["x"]) * w, float(p["y"]) * h
            d.ellipse([x - r, y - r, x + r, y + r], fill=(30, 190, 90))
    pts = res["points"]
    for a, b in (("shoulder_edge_l", "shoulder_edge_r"), ("neck_edge_l", "neck_edge_r"),
                 ("hip_edge_l", "hip_edge_r"), ("waist_edge_l", "waist_edge_r")):
        if a in pts and b in pts:
            d.line([pts[a][0] * w, pts[a][1] * h, pts[b][0] * w, pts[b][1] * h],
                   fill=(230, 40, 40), width=max(1, r // 2))
    for name, (fx, fy) in pts.items():
        x, y = fx * w, fy * h
        d.ellipse([x - r, y - r, x + r, y + r], fill=(230, 40, 40))
        d.text((x + r + 2, y - r), name, fill=(0, 0, 0))
    return flat


def main():
    from core.storage.supabase_client import supabase
    from tryon.vto_renderer import _get_with_retry

    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=KEJI)
    args = ap.parse_args()

    res = supabase.table("avatars").select("user_id, processed_photo_url, pose_keypoints") \
        .eq("user_id", args.user).limit(1).execute()
    if not res.data:
        print("No avatar for", args.user)
        return
    row = res.data[0]
    if not row.get("pose_keypoints"):
        print("This avatar has no body points")
        return

    img = Image.open(io.BytesIO(_get_with_retry(row["processed_photo_url"], "avatar processed")))
    result = measure(img, row["pose_keypoints"])

    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.user[:8]
    draw(img, row["pose_keypoints"], result).save(OUT / f"{tag}_outline.png")
    (OUT / f"{tag}_outline.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("\nSaved in", OUT)


if __name__ == "__main__":
    main()