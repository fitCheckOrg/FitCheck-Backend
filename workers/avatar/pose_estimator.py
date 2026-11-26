"""
Body pose on an avatar photo (MediaPipe PoseLandmarker, Tasks API).

Try-On Studio B10 (2026-11-26):
  * no temp file any more - the image goes to MediaPipe from memory (the old shared
    _tmp_pose_input.png could be overwritten by a second upload running at the same time);
  * the detector is used under a lock (one MediaPipe detector isn't safe across threads);
  * a transparent photo is placed on light grey first, so the model sees a clean
    background instead of whatever colour the transparent pixels happen to hold.
"""

import io
import threading

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image

from core.config.settings import settings
from shared.logger import get_logger

logger = get_logger(__name__)

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

BACKDROP = (232, 232, 234)          # light grey behind transparent pixels

_base_options = python.BaseOptions(model_asset_path=settings.POSE_MODEL_PATH)
_options = vision.PoseLandmarkerOptions(
    base_options=_base_options,
    running_mode=vision.RunningMode.IMAGE
)
_detector = vision.PoseLandmarker.create_from_options(_options)
_lock = threading.Lock()


def _rgb_array(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        flat = Image.new("RGB", rgba.size, BACKDROP)
        flat.paste(rgba, mask=rgba.split()[3])
        img = flat
    else:
        img = img.convert("RGB")
    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))


def estimate_pose(image_bytes: bytes) -> dict | None:
    """
    Normalized (0.0-1.0, fractions of THIS image) coordinates for all 33 body
    landmarks, plus MediaPipe's visibility and presence scores.
    MediaPipe's "left" is the person's left.

    Returns None on failure / no detection - pose data is an enhancement,
    never a blocking requirement for avatar creation.
    """
    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=_rgb_array(image_bytes))
        with _lock:
            result = _detector.detect(mp_image)

        if not result.pose_landmarks:
            logger.warning("Pose estimation - no pose detected in image")
            return None

        detected = result.pose_landmarks[0]  # first detected person
        landmarks = {}
        for name, point in zip(LANDMARK_NAMES, detected):
            landmarks[name] = {
                "x": round(point.x, 4),
                "y": round(point.y, 4),
                "visibility": round(point.visibility, 4),
                "presence": round(point.presence, 4)
            }

        logger.info("Pose estimation successful - %d landmarks detected", len(landmarks))
        return landmarks

    except Exception:
        logger.exception("Pose estimation failed")
        return None