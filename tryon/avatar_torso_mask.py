"""
Constructs a torso-specific region from four real skeletal
landmarks (left/right shoulder, left/right hip), intersected with
the avatar's actual alpha mask. Produces a visible overlay for
direct inspection — no left/right boundary extraction yet, no
correspondence mesh, no warping. Pure ROI validation.

If this looks anatomically correct, THEN we extract left/right
torso boundaries from within it. If not, that's real evidence
toward needing a dedicated segmentation model.

Run: python -m tryon.avatar_torso_mask
"""

import io
import os
import requests
import numpy as np
import cv2
from PIL import Image

ALPHA_THRESHOLD = 10
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
OUTPUT_DIR = "outputs"


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255, image


def build_torso_roi_mask(binary_mask, left_shoulder, right_shoulder, left_hip, right_hip, margin=10):
    """
    Builds a quadrilateral ROI from the four real pose landmarks
    (with a small margin so the torso's actual width isn't clipped
    right at the joint centers), then intersects it with the real
    alpha mask — so the result only contains pixels that are BOTH
    inside the anatomical torso region AND part of the real body
    silhouette.
    """
    h, w = binary_mask.shape

    # Order points as a proper quadrilateral: shoulder-L, shoulder-R, hip-R, hip-L
    quad = np.array([
        [left_shoulder[0] - margin, left_shoulder[1] - margin],
        [right_shoulder[0] + margin, right_shoulder[1] - margin],
        [right_hip[0] + margin, right_hip[1] + margin],
        [left_hip[0] - margin, left_hip[1] + margin],
    ], dtype=np.int32)

    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(roi_mask, quad, 255)

    torso_mask = cv2.bitwise_and(binary_mask, roi_mask)
    return torso_mask, quad


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    avatar_resp = requests.get(
        f"http://127.0.0.1:8000/api/avatar/me?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    avatar_bytes = requests.get(avatar_resp["processed_photo_url"]).content
    binary_mask, avatar_img = get_binary_mask(avatar_bytes)
    aw, ah = avatar_img.size

    pose_keypoints = avatar_resp["pose_keypoints"]
    left_shoulder = (pose_keypoints["left_shoulder"]["x"] * aw, pose_keypoints["left_shoulder"]["y"] * ah)
    right_shoulder = (pose_keypoints["right_shoulder"]["x"] * aw, pose_keypoints["right_shoulder"]["y"] * ah)
    left_hip = (pose_keypoints["left_hip"]["x"] * aw, pose_keypoints["left_hip"]["y"] * ah)
    right_hip = (pose_keypoints["right_hip"]["x"] * aw, pose_keypoints["right_hip"]["y"] * ah)

    print(f"left_shoulder={left_shoulder}  right_shoulder={right_shoulder}")
    print(f"left_hip={left_hip}  right_hip={right_hip}")

    torso_mask, quad = build_torso_roi_mask(binary_mask, left_shoulder, right_shoulder, left_hip, right_hip)

    # Visualize: original avatar, ROI quad outline, and the resulting 
    # intersected torso mask as a colored overlay
    canvas = avatar_img.convert("RGB").copy()
    overlay = np.array(canvas).copy()

    torso_pixels = torso_mask > 0
    overlay[torso_pixels] = [
        overlay[torso_pixels][:, 0] // 2,
        (overlay[torso_pixels][:, 1] // 2 + 127),
        overlay[torso_pixels][:, 2] // 2,
    ][0].astype(np.uint8) if False else overlay[torso_pixels]  # placeholder, fixed below

    # Simpler, correct overlay: blend green into torso pixels
    overlay = np.array(canvas).astype(np.int32)
    green_tint = np.zeros_like(overlay)
    green_tint[:, :, 1] = 255
    mask_3ch = np.stack([torso_pixels]*3, axis=-1)
    overlay = np.where(mask_3ch, (overlay * 0.5 + green_tint * 0.5).astype(np.int32), overlay)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    result = Image.fromarray(overlay)
    draw_canvas = result.copy()
    from PIL import ImageDraw
    draw = ImageDraw.Draw(draw_canvas)
    quad_pts = [tuple(p) for p in quad]
    draw.polygon(quad_pts, outline=(255, 255, 0), width=2)

    for pt, label in [(left_shoulder, "L_SH"), (right_shoulder, "R_SH"),
                        (left_hip, "L_HIP"), (right_hip, "R_HIP")]:
        x, y = pt
        draw.ellipse([x-6, y-6, x+6, y+6], outline=(255, 0, 0), width=2)
        draw.text((x+8, y-6), label, fill=(255, 0, 0))

    draw_canvas.save(f"{OUTPUT_DIR}/avatar_torso_mask.png")
    print(f"\nSaved: {OUTPUT_DIR}/avatar_torso_mask.png")


if __name__ == "__main__":
    main()
    