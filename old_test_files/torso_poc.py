"""
FitCheck — Torso-Region Transformation POC

Compares:
  A. Current approach — whole garment image, scaled to fit shoulder
     width (mirrors GarmentPositioning's actual logic)
  B. New approach — crop torso region using validated
     GarmentRepresentation, scale/position using avatar shoulder
     landmarks only

Standalone script. Does NOT touch Flutter, the database,
ClothingItem, or any API. Uses real, already-confirmed data from
tonight's testing — real avatar pose_keypoints, real garment with
a validated torso_region.

Run from inside FitCheckAI, venv activated:
    python torso_poc.py
"""

import io
import math
import requests
from PIL import Image

# ---- Real data — pulls live from your actual database ----
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
GARMENT_ITEM_ID = "5d19a29d-a7fa-4579-a1ba-2e374c3fd410"  # long sleeve t-shirt,
                                                             # confirmed valid
                                                             # representation tonight,
                                                             # no normalization needed

# Real, already-validated representation for this exact garment
# (confirmed tonight — reused here rather than re-calling GPT,
# keeping this a pure geometry test, not a new AI dependency)
GARMENT_TORSO_REGION = {"left_percent": 20, "right_percent": 80}

# Real confirmed config values
ANCHOR_OFFSET_RATIO = 0.08
SCALE_MULTIPLIER = 1.15


def fetch_avatar():
    """Pulls real avatar photo + pose_keypoints from the live API."""
    resp = requests.get(
        f"http://127.0.0.1:8000/api/avatar/me?user_id={AVATAR_USER_ID}"
    ).json()
    data = resp["data"]
    img_resp = requests.get(data["processed_photo_url"])
    avatar_img = Image.open(io.BytesIO(img_resp.content)).convert("RGBA")
    return avatar_img, data["pose_keypoints"]


def fetch_garment():
    """Pulls real garment photo from the live API."""
    resp = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{GARMENT_ITEM_ID}?user_id={AVATAR_USER_ID}"
    ).json()
    data = resp["data"]
    img_resp = requests.get(data["clean_image_url"])
    garment_img = Image.open(io.BytesIO(img_resp.content)).convert("RGBA")
    return garment_img


def get_shoulder_geometry(avatar_img, keypoints):
    w, h = avatar_img.size
    left = keypoints["left_shoulder"]
    right = keypoints["right_shoulder"]
    lx, ly = left["x"] * w, left["y"] * h
    rx, ry = right["x"] * w, right["y"] * h
    distance = math.hypot(rx - lx, ry - ly)
    mid_x, mid_y = (lx + rx) / 2, (ly + ry) / 2
    return distance, mid_x, mid_y


def build_current_approach(avatar_img, garment_img, shoulder_dist, mid_x, mid_y):
    """Mirrors GarmentPositioning's existing logic — whole garment,
    scaled to shoulder width."""
    result = avatar_img.copy()
    target_width = shoulder_dist * SCALE_MULTIPLIER
    aspect = garment_img.height / garment_img.width
    target_height = target_width * aspect
    resized = garment_img.resize((int(target_width), int(target_height)), Image.LANCZOS)
    paste_x = int(mid_x - target_width / 2)
    paste_y = int(mid_y - target_height * ANCHOR_OFFSET_RATIO)
    result.paste(resized, (paste_x, paste_y), resized)
    return result


def build_torso_representation_approach(avatar_img, garment_img, torso_region, shoulder_dist, mid_x, mid_y):
    """New approach — crop just the validated torso region, scale
    to shoulder width only (not whole-garment width)."""
    gw, gh = garment_img.size
    left_px = int(gw * torso_region["left_percent"] / 100)
    right_px = int(gw * torso_region["right_percent"] / 100)
    torso_crop = garment_img.crop((left_px, 0, right_px, gh))

    result = avatar_img.copy()
    target_width = shoulder_dist * SCALE_MULTIPLIER
    aspect = torso_crop.height / torso_crop.width
    target_height = target_width * aspect
    resized = torso_crop.resize((int(target_width), int(target_height)), Image.LANCZOS)
    paste_x = int(mid_x - target_width / 2)
    paste_y = int(mid_y - target_height * ANCHOR_OFFSET_RATIO)
    result.paste(resized, (paste_x, paste_y), resized)
    return result


def main():
    print("Fetching real avatar and garment...")
    avatar_img, pose_keypoints = fetch_avatar()
    garment_img = fetch_garment()

    shoulder_dist, mid_x, mid_y = get_shoulder_geometry(avatar_img, pose_keypoints)
    print(f"Shoulder distance: {shoulder_dist:.1f}px, midpoint: ({mid_x:.1f}, {mid_y:.1f})")

    current = build_current_approach(avatar_img, garment_img, shoulder_dist, mid_x, mid_y)
    current.save("poc_output_current_rectangle.png")
    print("Saved: poc_output_current_rectangle.png  (whole-garment approach)")

    torso_version = build_torso_representation_approach(
        avatar_img, garment_img, GARMENT_TORSO_REGION, shoulder_dist, mid_x, mid_y
    )
    torso_version.save("poc_output_torso_representation.png")
    print("Saved: poc_output_torso_representation.png  (torso-only approach)")

    print("\nOpen both PNGs side by side and compare.")


if __name__ == "__main__":
    main()