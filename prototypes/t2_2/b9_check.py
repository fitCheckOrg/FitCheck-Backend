"""
Try-On Studio B9 + B10 check (2026-11-26) - measures the real files, no provider calls, $0.

Answers:
  C1 (a)  do wardrobe image_width / image_height describe the clean cut-out file?
  C5 (a)  is processed_photo_url background-removed + transparent?
  C5 (c)  do photo_width / photo_height match the processed file?
  C6      does a render line up with the avatar photo (original or processed)?
And draws the stored body points over the photos (B10 debug overlay):
  Keji's processed photo (what the Studio stage shows), Keji's original photo,
  Keji's latest render, and the two P4b synthetic people.

Run from the FitCheckAI folder (venv active):
    python -m prototypes.t2_2.b9_check

Outputs: prototypes/t2_2/out/b9/report.json + *_points.png (local only, gitignored).
"""

import io
import json

from PIL import Image, ImageDraw

from core.storage.s3 import storage
from core.storage.supabase_client import supabase
from prototypes.t2_2.p1_modes import OUT, USER_ID
from tryon.vto_renderer import _get_with_retry
from workers.avatar.pose_estimator import estimate_pose

B9 = OUT / "b9"

# Studio names -> MediaPipe names (MediaPipe "left" = the person's left).
POINTS = [
    ("nose", "nose"),
    ("shoulder_l", "left_shoulder"), ("shoulder_r", "right_shoulder"),
    ("elbow_l", "left_elbow"), ("elbow_r", "right_elbow"),
    ("wrist_l", "left_wrist"), ("wrist_r", "right_wrist"),
    ("hip_l", "left_hip"), ("hip_r", "right_hip"),
    ("knee_l", "left_knee"), ("knee_r", "right_knee"),
    ("ankle_l", "left_ankle"), ("ankle_r", "right_ankle"),
]
BONES = [("shoulder_l", "shoulder_r"), ("shoulder_l", "elbow_l"), ("elbow_l", "wrist_l"),
         ("shoulder_r", "elbow_r"), ("elbow_r", "wrist_r"), ("shoulder_l", "hip_l"),
         ("shoulder_r", "hip_r"), ("hip_l", "hip_r"), ("hip_l", "knee_l"), ("knee_l", "ankle_l"),
         ("hip_r", "knee_r"), ("knee_r", "ankle_r")]


def _load(url: str):
    data = _get_with_retry(url, "image")
    return data, Image.open(io.BytesIO(data))


def _describe(what: str, im: Image.Image, data: bytes) -> dict:
    has_alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
    transparent = None
    if has_alpha:
        low, _ = im.convert("RGBA").split()[3].getextrema()
        transparent = low < 255
    return {"what": what, "size": list(im.size), "ratio_w_h": round(im.size[0] / im.size[1], 4),
            "mode": im.mode, "format": im.format, "has_alpha": has_alpha,
            "has_transparent_pixels": transparent, "kb": len(data) // 1024}


def _studio(lm: dict | None) -> dict | None:
    """MediaPipe 33 points -> the Studio shape (subset, fractions of the image)."""
    if not lm:
        return None
    out = {}
    for name, mp_name in POINTS:
        p = lm.get(mp_name)
        if p and float(p.get("visibility", 1)) >= 0.5:
            out[name] = [round(float(p["x"]), 4), round(float(p["y"]), 4)]
    if "shoulder_l" in out and "shoulder_r" in out:
        out["neck"] = [round((out["shoulder_l"][0] + out["shoulder_r"][0]) / 2, 4),
                       round((out["shoulder_l"][1] + out["shoulder_r"][1]) / 2, 4)]
    return out


def _overlay(im: Image.Image, points: dict | None, name: str) -> None:
    canvas = Image.new("RGBA", im.size, (225, 225, 228, 255))
    canvas.alpha_composite(im.convert("RGBA"))
    d = ImageDraw.Draw(canvas)
    w, h = canvas.size
    r = max(5, w // 110)
    if points:
        for a, b in BONES:
            if a in points and b in points:
                d.line([(points[a][0] * w, points[a][1] * h), (points[b][0] * w, points[b][1] * h)],
                       fill=(255, 196, 0, 255), width=max(2, r // 2))
        for key, (x, y) in points.items():
            cx, cy = x * w, y * h
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0, 200, 90, 255), outline=(0, 0, 0, 255))
            d.text((cx + r + 2, cy - r), key, fill=(0, 0, 0, 255))
    else:
        d.text((10, 10), "NO POSE FOUND", fill=(220, 0, 0, 255))
    canvas.convert("RGB").save(B9 / f"{name}_points.png")


def _mid(points: dict, a: str, b: str):
    if points and a in points and b in points:
        return [round((points[a][0] + points[b][0]) / 2, 4), round((points[a][1] + points[b][1]) / 2, 4)]
    return None


def _frame(points: dict | None) -> dict:
    """Where the body sits in the picture (fractions): shoulders, hips, ankles, body height."""
    sh, hp, an = _mid(points, "shoulder_l", "shoulder_r"), _mid(points, "hip_l", "hip_r"), \
        _mid(points, "ankle_l", "ankle_r")
    tall = round(an[1] - points["nose"][1], 4) if an and points and "nose" in points else None
    return {"shoulders_mid": sh, "hips_mid": hp, "ankles_mid": an, "nose_to_ankles": tall}


def main():
    B9.mkdir(parents=True, exist_ok=True)
    report = {}

    # ---- avatar (C5)
    av = supabase.table("avatars").select(
        "avatar_version, original_photo_url, processed_photo_url, photo_width, photo_height, pose_keypoints")\
        .eq("user_id", USER_ID).limit(1).execute().data[0]
    o_bytes, o_img = _load(av["original_photo_url"])
    p_bytes, p_img = _load(av["processed_photo_url"])
    stored = _studio(av.get("pose_keypoints"))
    fresh_original = _studio(estimate_pose(o_bytes))
    report["avatar"] = {
        "avatar_version": av["avatar_version"],
        "original": _describe("original_photo_url (what FASHN gets)", o_img, o_bytes),
        "processed": _describe("processed_photo_url (what the stage shows)", p_img, p_bytes),
        "db_photo_w_h": [av.get("photo_width"), av.get("photo_height")],
        "db_matches_processed": [av.get("photo_width"), av.get("photo_height")] == list(p_img.size),
        "stored_pose_keypoints": bool(av.get("pose_keypoints")),
        "stored_points_on_processed": stored,
        "frame_processed": _frame(stored),
        "frame_original": _frame(fresh_original),
    }
    _overlay(p_img, stored, "keji_processed")
    _overlay(o_img, fresh_original, "keji_original")

    # ---- latest single render (C6)
    rows = supabase.table("tryon_renders").select("s3_key, renderer_version, completed_at")\
        .eq("user_id", USER_ID).eq("status", "completed").eq("renderer_version", "v1")\
        .eq("avatar_version", av["avatar_version"]).order("completed_at", desc=True).limit(1).execute().data
    if rows and rows[0].get("s3_key"):
        r_bytes, r_img = _load(storage.get_url(rows[0]["s3_key"]))
        r_points = _studio(estimate_pose(r_bytes))
        report["render"] = {"file": _describe("latest single render", r_img, r_bytes),
                            "frame": _frame(r_points)}
        _overlay(r_img, r_points, "keji_render")
    else:
        report["render"] = "no completed single render for the current avatar"

    # ---- wardrobe cut-outs (C1 a, c)
    items = supabase.table("closet_items").select("item_id, item_type, image_width, image_height, clean_image_url")\
        .eq("user_id", USER_ID).eq("is_archived", False).order("created_at", desc=True).limit(5).execute().data or []
    report["cutouts"] = []
    for it in items:
        c_bytes, c_img = _load(it["clean_image_url"])
        d = _describe(it.get("item_type") or "item", c_img, c_bytes)
        d["db_w_h"] = [it.get("image_width"), it.get("image_height")]
        d["db_matches_file"] = d["db_w_h"] == d["size"]
        report["cutouts"].append(d)

    # ---- synthetic people (B10 overlay check, no real person)
    report["synthetic"] = {}
    for name in ("s1_Y", "s2_Y"):
        path = OUT / "p4b" / f"{name}.png"
        if path.exists():
            data = path.read_bytes()
            pts = _studio(estimate_pose(data))
            report["synthetic"][name] = {"points_found": len(pts or {}), "frame": _frame(pts)}
            _overlay(Image.open(io.BytesIO(data)), pts, f"synth_{name}")

    (B9 / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nOverlays + report in {B9}")


if __name__ == "__main__":
    main()