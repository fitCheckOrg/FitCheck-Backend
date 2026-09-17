import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from shared.logger import get_logger
from core.config.settings import settings

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

_base_options = python.BaseOptions(model_asset_path=settings.POSE_MODEL_PATH)
_options = vision.PoseLandmarkerOptions(
    base_options=_base_options,
    running_mode=vision.RunningMode.IMAGE
)
_detector = vision.PoseLandmarker.create_from_options(_options)


def estimate_pose(image_bytes: bytes) -> dict | None:
    """
    Runs MediaPipe's PoseLandmarker (Tasks API) on an image, returns
    normalized (0.0-1.0) coordinates for all 33 body landmarks, plus
    MediaPipe's own visibility and presence confidence scores.

    Returns None on failure/no-detection — pose data is an
    enhancement, never a blocking requirement for avatar creation.
    """
    try:
        with open("_tmp_pose_input.png", "wb") as f:
            f.write(image_bytes)
        mp_image = mp.Image.create_from_file("_tmp_pose_input.png")

        result = _detector.detect(mp_image)

        if not result.pose_landmarks:
            logger.warning("Pose estimation — no pose detected in image")
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

        logger.info("Pose estimation successful — %d landmarks detected", len(landmarks))
        return landmarks

    except Exception:
        logger.exception("Pose estimation failed")
        return None