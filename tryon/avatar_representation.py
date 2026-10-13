"""
AvatarRepresentation — the representation layer confirmed feasible
this session. Deliberately boring: a thin wrapper holding real
SCHP body-part masks and real MediaPipe pose keypoints together,
with zero inferred geometry. No torso_left/torso_right guesses, no
corridors, no thresholds.

torso_mask etc. are real per-pixel boolean masks — not coordinates,
not approximations.

Run standalone: python -m tryon.avatar_representation
(fetches the golden avatar and prints a sanity summary)
"""

import functools
import io
import logging
import numpy as np
import requests
from PIL import Image
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor

logger = logging.getLogger(__name__)

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
SCHP_MODEL_ID = "pirocheto/schp-pascal-7"

# SCHP Pascal-7 label ids, confirmed from the model card
LABEL_BACKGROUND = 0
LABEL_HEAD = 1
LABEL_TORSO = 2
LABEL_UPPER_ARMS = 3
LABEL_LOWER_ARMS = 4
LABEL_UPPER_LEGS = 5
LABEL_LOWER_LEGS = 6


class AvatarRepresentation:
    """
    image: PIL.Image, the processed avatar photo
    body_part_masks: dict[str, np.ndarray[bool]] — one full-resolution
        boolean mask per SCHP class, keyed by name
    pose_keypoints: dict — the raw, already-validated MediaPipe
        keypoints (normalized 0-1 fractions), untouched

    Convenience accessors below just index into body_part_masks /
    pose_keypoints — they don't compute anything new.
    """

    def __init__(self, image, body_part_masks, pose_keypoints):
        self.image = image
        self.body_part_masks = body_part_masks
        self.pose_keypoints = pose_keypoints

    @property
    def torso_mask(self):
        return self.body_part_masks["torso"]

    @property
    def upper_arms_mask(self):
        return self.body_part_masks["upper_arms"]

    @property
    def lower_arms_mask(self):
        return self.body_part_masks["lower_arms"]

    @property
    def upper_legs_mask(self):
        return self.body_part_masks["upper_legs"]

    @property
    def lower_legs_mask(self):
        return self.body_part_masks["lower_legs"]

    @property
    def head_mask(self):
        return self.body_part_masks["head"]

    def _kp(self, name):
        aw, ah = self.image.size
        p = self.pose_keypoints[name]
        return (p["x"] * aw, p["y"] * ah)

    @property
    def left_shoulder(self):
        return self._kp("left_shoulder")

    @property
    def right_shoulder(self):
        return self._kp("right_shoulder")

    @property
    def left_elbow(self):
        return self._kp("left_elbow")

    @property
    def right_elbow(self):
        return self._kp("right_elbow")

    @property
    def left_wrist(self):
        return self._kp("left_wrist")

    @property
    def right_wrist(self):
        return self._kp("right_wrist")

    @property
    def left_hip(self):
        return self._kp("left_hip")

    @property
    def right_hip(self):
        return self._kp("right_hip")
    
    @property
    def left_underarm(self):
        if not hasattr(self, "_left_underarm"):
            self._compute_underarms()
        return self._left_underarm

    @property
    def right_underarm(self):
        if not hasattr(self, "_right_underarm"):
            self._compute_underarms()
        return self._right_underarm

    @property
    def underarm_result(self):
        """The UnderarmLandmarkResult (see resolve_underarms)."""
        return self.resolve_underarms()

    def resolve_underarms(self):
        """
        Resolve the underarm landmarks once and return the full
        UnderarmLandmarkResult: per-landmark source, both estimates, method
        and landmark revision. Never raises for bad data - an unresolvable
        photo comes back as an UNRESOLVED result. (left_underarm and
        right_underarm still raise the 422 for that case.)
        """
        if not hasattr(self, "_underarm_result"):
            from tryon.underarm_landmark_resolver import resolve_underarm_landmarks
            self._underarm_result = resolve_underarm_landmarks(
                self.image,
                self.left_shoulder,
                self.right_shoulder,
                detector_left=self._detector_underarm("left"),
                detector_right=self._detector_underarm("right"),
            )
        return self._underarm_result

    def _detector_underarm(self, side):
        """
        Mask-based detector underarm for one side as (x, y), or None.
        A missing row or an empty mask row means "no detector estimate for
        this side"; so does an error raised while running the detector, which
        is logged rather than swallowed. The x rule is the one
        _compute_underarms always used (outer edge of the torso mask on that
        row, within that image half).
        """
        # Import outside the try: a broken/missing detector module is a
        # deployment bug and must fail loudly, not silently fall back.
        from tryon.avatar_torso_landmarks_test import find_semantic_underarm
        try:
            aw = self.image.size[0]
            found = find_semantic_underarm(self.torso_mask, self.upper_arms_mask, aw, side)
            row = found[0] if found else None
            if row is None:
                return None
            row = int(row)
            if not 0 <= row < self.torso_mask.shape[0]:
                return None
            if side == "left":
                xs = np.where(self.torso_mask[row][:aw // 2])[0]
                return (int(xs.min()), row) if xs.size else None
            xs = np.where(self.torso_mask[row][aw // 2:])[0] + aw // 2
            return (int(xs.max()), row) if xs.size else None
        except Exception:
            logger.exception("underarm detector failed (%s side); treating as no estimate", side)
            return None

    def _compute_underarms(self):
        from shared.exceptions import FitCheckException

        result = self.resolve_underarms()
        if not result.resolved:
            raise FitCheckException(
                "Could not locate the underarm points on this avatar photo.",
                code="UNDERARM_NOT_DETECTED",
                status_code=422,
            )
        # Same (int x, int y) tuples the rest of the pipeline already consumes.
        self._left_underarm = (int(round(result.left.x)), int(round(result.left.y)))
        self._right_underarm = (int(round(result.right.x)), int(round(result.right.y)))


@functools.lru_cache(maxsize=1)
def _load_schp():
    """
    Load the SCHP processor and ONNX session once per process and reuse them.
    (Previously every segmentation call re-read the model from disk, which is
    fine for a script and too slow for an upload endpoint.) Failures are not
    cached: a failed load is retried on the next call.
    """
    onnx_path = hf_hub_download(repo_id=SCHP_MODEL_ID, filename="onnx/schp-pascal-7.onnx")
    hf_hub_download(repo_id=SCHP_MODEL_ID, filename="onnx/schp-pascal-7.onnx.data")
    processor = AutoImageProcessor.from_pretrained(SCHP_MODEL_ID, trust_remote_code=True)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return processor, session, session.get_inputs()[0].name


def _run_schp_segmentation(pil_image):
    """
    Exactly the inference path validated in human_parsing_test.py -
    ONNX, no torch/schp package, no compiler dependency.
    """
    processor, session, input_name = _load_schp()

    aw, ah = pil_image.size
    inputs = processor(images=pil_image, return_tensors="np")
    pixel_values = inputs["pixel_values"].astype(np.float32)

    outputs = session.run(None, {input_name: pixel_values})
    logits = outputs[0]

    import cv2
    logits_resized = np.zeros((1, logits.shape[1], ah, aw), dtype=np.float32)
    for c in range(logits.shape[1]):
        logits_resized[0, c] = cv2.resize(logits[0, c], (aw, ah), interpolation=cv2.INTER_LINEAR)

    parsing_mask = np.argmax(logits_resized[0], axis=0)
    return parsing_mask


def build_avatar_representation_from_image(image, pose_keypoints):
    """
    Build a real AvatarRepresentation from an RGB PIL image and its pose
    keypoints, both already in memory (no database or network access).
    This is the segmentation half of build_avatar_representation, split out
    so the upload pipeline can run the same check on the photo it just
    processed. Raises ValueError if pose_keypoints are missing.
    """
    if not pose_keypoints:
        raise ValueError("No pose_keypoints available for this avatar - cannot build representation.")

    parsing_mask = _run_schp_segmentation(image)

    body_part_masks = {
        "head": parsing_mask == LABEL_HEAD,
        "torso": parsing_mask == LABEL_TORSO,
        "upper_arms": parsing_mask == LABEL_UPPER_ARMS,
        "lower_arms": parsing_mask == LABEL_LOWER_ARMS,
        "upper_legs": parsing_mask == LABEL_UPPER_LEGS,
        "lower_legs": parsing_mask == LABEL_LOWER_LEGS,
    }
    return AvatarRepresentation(image, body_part_masks, pose_keypoints)


def build_avatar_representation(avatar_user_id=AVATAR_USER_ID):
    """
    Fetches the current avatar, runs SCHP, and returns a real
    AvatarRepresentation. No fallbacks, no synthetic geometry - if
    pose_keypoints are missing, this will raise rather than guess.
    """
    from core.storage.supabase_client import supabase
    from shared.exceptions import AvatarNotFoundError

    result = supabase.table("avatars")\
        .select("avatar_id, avatar_version, processed_photo_url, pose_keypoints")\
        .eq("user_id", str(avatar_user_id))\
        .maybe_single()\
        .execute()

    if not result or not result.data:
        raise AvatarNotFoundError()

    avatar_resp = result.data
    avatar_bytes = requests.get(avatar_resp["processed_photo_url"]).content
    image = Image.open(io.BytesIO(avatar_bytes)).convert("RGB")

    rep = build_avatar_representation_from_image(image, avatar_resp["pose_keypoints"])
    rep.avatar_id = avatar_resp["avatar_id"]
    rep.avatar_version = avatar_resp["avatar_version"]
    rep.processed_photo_url = avatar_resp["processed_photo_url"]
    return rep


if __name__ == "__main__":
    rep = build_avatar_representation()
    print(f"Image size: {rep.image.size}")
    print(f"Torso mask pixel count: {rep.torso_mask.sum()}")
    print(f"Upper arms mask pixel count: {rep.upper_arms_mask.sum()}")
    print(f"Lower arms mask pixel count: {rep.lower_arms_mask.sum()}")
    print(f"Left shoulder (px): {rep.left_shoulder}")
    print(f"Right shoulder (px): {rep.right_shoulder}")
    print(f"Left hip (px): {rep.left_hip}")
    print(f"Right hip (px): {rep.right_hip}")
    print("\n>>> AvatarRepresentation built successfully - masks and keypoints held together, no inferred geometry.")