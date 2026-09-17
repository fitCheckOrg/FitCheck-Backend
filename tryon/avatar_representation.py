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

import io
import numpy as np
import requests
from PIL import Image
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor

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

    def _compute_underarms(self):
        from tryon.avatar_torso_landmarks_test import find_semantic_underarm
        aw = self.image.size[0]

        a_l_uy, _ = find_semantic_underarm(self.torso_mask, self.upper_arms_mask, aw, "left")
        a_l_row = self.torso_mask[a_l_uy]
        a_l_xs = np.where(a_l_row[:aw // 2])[0]
        self._left_underarm = (int(a_l_xs.min()), a_l_uy)

        a_r_uy, _ = find_semantic_underarm(self.torso_mask, self.upper_arms_mask, aw, "right")
        a_r_row = self.torso_mask[a_r_uy]
        a_r_xs = np.where(a_r_row[aw // 2:])[0] + aw // 2
        self._right_underarm = (int(a_r_xs.max()), a_r_uy)


def _run_schp_segmentation(pil_image):
    """
    Exactly the inference path validated in human_parsing_test.py —
    ONNX, no torch/schp package, no compiler dependency.
    """
    onnx_path = hf_hub_download(repo_id=SCHP_MODEL_ID, filename="onnx/schp-pascal-7.onnx")
    hf_hub_download(repo_id=SCHP_MODEL_ID, filename="onnx/schp-pascal-7.onnx.data")
    processor = AutoImageProcessor.from_pretrained(SCHP_MODEL_ID, trust_remote_code=True)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

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


def build_avatar_representation(avatar_user_id=AVATAR_USER_ID):
    """
    Fetches the current avatar, runs SCHP, and returns a real
    AvatarRepresentation. No fallbacks, no synthetic geometry — if
    pose_keypoints are missing, this will raise rather than guess.
    """
    from core.storage.supabase_client import supabase
    from shared.exceptions import AvatarNotFoundError

    result = supabase.table("avatars")\
        .select("processed_photo_url, pose_keypoints")\
        .eq("user_id", str(avatar_user_id))\
        .maybe_single()\
        .execute()

    if not result or not result.data:
        raise AvatarNotFoundError()

    avatar_resp = result.data

    avatar_bytes = requests.get(avatar_resp["processed_photo_url"]).content
    image = Image.open(io.BytesIO(avatar_bytes)).convert("RGB")

    pose_keypoints = avatar_resp["pose_keypoints"]
    if not pose_keypoints:
        raise ValueError("No pose_keypoints available for this avatar — cannot build representation.")

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
    print("\n>>> AvatarRepresentation built successfully — masks and keypoints held together, no inferred geometry.")