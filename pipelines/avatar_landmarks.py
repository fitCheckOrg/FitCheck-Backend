"""
Try-On Studio B10 / B10b (2026-11-26): the avatar's body points in the shape the app
uses (S1 in BACKEND_STUDIO_WORKPLAN_2026-11-26.md).

    "landmarks": {
      "version": 2,
      "image_w": 736, "image_h": 1308,        # = processed_photo_url's size
      "confidence": 0.99,                     # mean visibility of the joint points
      "points": {"nose": [x, y], "neck": [x, y], "shoulder_edge_l": [x, y], ...}
    }

* x, y = fractions (0-1, top-left origin) of the exact image at processed_photo_url.
* _l / _r = the PERSON's left / right (on a front-facing photo, _l is on the image's right).
* Joint points (MediaPipe, centre of the joint): nose, shoulder_l/r, elbow_l/r, wrist_l/r,
  hip_l/r, knee_l/r, ankle_l/r.
* Outline points (B10b, measured from the photo's cut-out - where clothes sit):
  neck, neck_edge_l/r, shoulder_edge_l/r, shoulder_top_l/r, waist_edge_l/r, hip_edge_l/r.
* version 2 change: `neck` is now MEASURED where the neck meets the shoulders
  (v1 put it halfway between the shoulder joints, which sits on the collarbones).
* A point that couldn't be seen or measured is left out - never guessed.
  No joints at all -> landmarks: null.
* Only ever returned to the avatar's owner (GET /avatar/me, the upload response).
"""

LANDMARKS_VERSION = 2
MIN_VISIBILITY = 0.5

JOINTS = [
    ("nose", "nose"),
    ("shoulder_l", "left_shoulder"), ("shoulder_r", "right_shoulder"),
    ("elbow_l", "left_elbow"), ("elbow_r", "right_elbow"),
    ("wrist_l", "left_wrist"), ("wrist_r", "right_wrist"),
    ("hip_l", "left_hip"), ("hip_r", "right_hip"),
    ("knee_l", "left_knee"), ("knee_r", "right_knee"),
    ("ankle_l", "left_ankle"), ("ankle_r", "right_ankle"),
]

OUTLINE = [
    "neck", "neck_edge_l", "neck_edge_r",
    "shoulder_edge_l", "shoulder_edge_r", "shoulder_top_l", "shoulder_top_r",
    "waist_edge_l", "waist_edge_r", "hip_edge_l", "hip_edge_r",
]


def _xy(p):
    try:
        x, y = float(p[0]), float(p[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    return [round(x, 4), round(y, 4)] if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 else None


def landmarks_for(avatar: dict | None) -> dict | None:
    if not avatar:
        return None
    raw = avatar.get("pose_keypoints") or None
    w, h = avatar.get("photo_width"), avatar.get("photo_height")
    if not raw or not w or not h:
        return None

    points, seen = {}, []
    for name, mp_name in JOINTS:
        p = raw.get(mp_name)
        if not p:
            continue
        try:
            vis = float(p.get("visibility", 1.0))
            xy = _xy((p["x"], p["y"]))
        except (KeyError, TypeError, ValueError):
            continue
        if xy is None or vis < MIN_VISIBILITY:
            continue
        points[name] = xy
        seen.append(vis)

    if not points:
        return None

    outline = avatar.get("body_outline") or {}
    for name in OUTLINE:
        if name in outline:
            xy = _xy(outline[name])
            if xy is not None:
                points[name] = xy

    return {
        "version": LANDMARKS_VERSION,
        "image_w": int(w),
        "image_h": int(h),
        "confidence": round(sum(seen) / len(seen), 2),
        "points": points,
    }


def with_landmarks(avatar) -> dict:
    """The avatar as the API returns it to its owner, plus `landmarks`."""
    if isinstance(avatar, dict):
        out = dict(avatar)
    elif hasattr(avatar, "model_dump"):
        out = avatar.model_dump()
    else:
        out = dict(avatar)
    out["landmarks"] = landmarks_for(out)
    return out