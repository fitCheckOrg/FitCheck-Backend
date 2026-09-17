"""
Step 3 of the reset architecture — ONNX version, avoiding the
schp PyPI package's CUDA/C++ compilation requirement entirely.
ONNX Runtime ships prebuilt, no compiler or CUDA toolkit needed.

Setup: pip install onnxruntime huggingface_hub --break-system-packages

Run: python -m tryon.human_parsing_test
"""

import io
import os
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor

from workers.avatar.geometry_derivation import derive_avatar_geometry

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
OUTPUT_DIR = "outputs"
MODEL_ID = "pirocheto/schp-pascal-7"

LABELS = ["background", "head", "torso", "upper_arms", "lower_arms", "upper_legs", "lower_legs"]
COLORS = [
    (0, 0, 0), (255, 220, 180), (0, 200, 0), (255, 0, 0),
    (255, 120, 0), (0, 100, 255), (0, 200, 255),
]
TORSO_LABEL_ID = 2
ARM_LABEL_IDS = [3, 4]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Downloading ONNX model + processor for {MODEL_ID}...")
    onnx_path = hf_hub_download(repo_id=MODEL_ID, filename="onnx/schp-pascal-7.onnx")
    hf_hub_download(repo_id=MODEL_ID, filename="onnx/schp-pascal-7.onnx.data")  # external weights, must sit alongside the .onnx file
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print(f"ONNX model loaded. Input: {input_name}")

    print("\nFetching golden avatar...")
    avatar_resp = requests.get(
        f"http://127.0.0.1:8000/api/avatar/me?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    print(f"avatar_version: {avatar_resp.get('avatar_version')}")

    avatar_bytes = requests.get(avatar_resp["processed_photo_url"]).content
    pil_image = Image.open(io.BytesIO(avatar_bytes)).convert("RGB")
    aw, ah = pil_image.size

    print("\nRunning SCHP Pascal-7 segmentation (ONNX)...")
    inputs = processor(images=pil_image, return_tensors="np")
    pixel_values = inputs["pixel_values"].astype(np.float32)

    outputs = session.run(None, {input_name: pixel_values})
    logits = outputs[0]

    logits_t = np.transpose(logits, (0, 1, 2, 3))
    logits_resized = np.zeros((1, logits.shape[1], ah, aw), dtype=np.float32)
    for c in range(logits.shape[1]):
        logits_resized[0, c] = cv2.resize(logits[0, c], (aw, ah), interpolation=cv2.INTER_LINEAR)

    parsing_mask = np.argmax(logits_resized[0], axis=0)

    unique_labels = np.unique(parsing_mask)
    print(f"\nLabels found: {unique_labels.tolist()}")
    for label_id in unique_labels:
        label_name = LABELS[label_id] if label_id < len(LABELS) else f"unknown({label_id})"
        pixel_count = np.sum(parsing_mask == label_id)
        pct = pixel_count / parsing_mask.size * 100
        print(f"  {label_id} ({label_name}): {pixel_count} px ({pct:.1f}%)")

    colored = np.zeros((ah, aw, 3), dtype=np.uint8)
    for label_id in unique_labels:
        color = COLORS[label_id] if label_id < len(COLORS) else (255, 0, 255)
        colored[parsing_mask == label_id] = color

    original_np = np.array(pil_image)
    side_by_side = np.concatenate([original_np, colored], axis=1)
    Image.fromarray(side_by_side).save(f"{OUTPUT_DIR}/avatar_human_parsing_test.png")
    print(f"\nSaved: {OUTPUT_DIR}/avatar_human_parsing_test.png")

    torso_mask = (parsing_mask == TORSO_LABEL_ID).astype(np.uint8) * 255
    arm_mask = np.isin(parsing_mask, ARM_LABEL_IDS).astype(np.uint8) * 255
    torso_contours, _ = cv2.findContours(torso_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = original_np.copy()
    arm_bool = arm_mask > 0
    overlay[arm_bool] = (overlay[arm_bool] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)

    overlay_img = Image.fromarray(overlay)
    draw = ImageDraw.Draw(overlay_img)
    for contour in torso_contours:
        pts = [(int(p[0][0]), int(p[0][1])) for p in contour]
        if len(pts) > 2:
            draw.line(pts + [pts[0]], fill=(0, 255, 0), width=3)

    pose_keypoints = avatar_resp["pose_keypoints"]
    avatar_geometry = derive_avatar_geometry(pose_keypoints)
    landmark_points = [
        (pose_keypoints["left_shoulder"]["x"] * aw, pose_keypoints["left_shoulder"]["y"] * ah, "L_shoulder"),
        (pose_keypoints["right_shoulder"]["x"] * aw, pose_keypoints["right_shoulder"]["y"] * ah, "R_shoulder"),
        (pose_keypoints["left_hip"]["x"] * aw, pose_keypoints["left_hip"]["y"] * ah, "L_hip"),
        (pose_keypoints["right_hip"]["x"] * aw, pose_keypoints["right_hip"]["y"] * ah, "R_hip"),
        (avatar_geometry.left_underarm.x_pct / 100 * aw, avatar_geometry.left_underarm.y_pct / 100 * ah, "L_UA"),
        (avatar_geometry.right_underarm.x_pct / 100 * aw, avatar_geometry.right_underarm.y_pct / 100 * ah, "R_UA"),
    ]
    for x, y, label in landmark_points:
        draw.ellipse([x-6, y-6, x+6, y+6], outline=(255, 255, 0), width=2)
        draw.text((x+8, y-6), label, fill=(255, 255, 0))

    overlay_img.save(f"{OUTPUT_DIR}/avatar_human_parsing_overlay.png")
    print(f"Saved: {OUTPUT_DIR}/avatar_human_parsing_overlay.png")
    print(f"\n>>> THE REAL QUESTION: does the green torso outline correspond to "
          f"the actual anatomical torso — excluding arms — on THIS avatar?")


if __name__ == "__main__":
    main()