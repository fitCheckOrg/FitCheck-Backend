"""
First correspondence visualization. Aligns the garment to the
avatar using ONLY the two semantic underarm anchor pairs — no hip
correspondence, no width matching, no deformation. The garment's
own shape is preserved exactly; only a single uniform scale +
translation (derived from the underarm-to-underarm distance) is
applied, so we can see honestly whether the two representations
agree once aligned at the one correspondence we've actually earned.

Run: python -m tryon.correspondence_anchors_test
"""

import io
import os
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw

from tryon.avatar_representation import build_avatar_representation
from tryon.avatar_torso_landmarks_test import find_semantic_underarm

GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
ALPHA_THRESHOLD = 10
MIN_DEFECT_DEPTH = 1000
OUTPUT_DIR = "outputs"


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD), image


def find_garment_underarms(mask, w_img):
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)
    hull_indices = cv2.convexHull(main_contour, returnPoints=False)
    defects = cv2.convexityDefects(main_contour, hull_indices)
    defects_flat = defects.reshape(-1, 4)

    left_candidates, right_candidates = [], []
    for i in range(len(defects_flat)):
        _, _, far_idx, depth = defects_flat[i]
        depth, far_idx = int(depth), int(far_idx)
        if depth <= MIN_DEFECT_DEPTH:
            continue
        far_pt = main_contour[far_idx][0]
        x_pct = far_pt[0] / w_img * 100
        (left_candidates if x_pct < 50 else right_candidates).append((depth, far_idx))
    left_candidates.sort(reverse=True)
    right_candidates.sort(reverse=True)

    left = tuple(int(v) for v in main_contour[left_candidates[0][1]][0])
    right = tuple(int(v) for v in main_contour[right_candidates[0][1]][0])
    return left, right, main_contour


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Fetching golden garment...")
    garment_resp = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{GOLDEN_GARMENT_ID}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    garment_bytes = requests.get(garment_resp["clean_image_url"]).content
    garment_mask, garment_img = get_binary_mask(garment_bytes)
    gw, gh = garment_img.size

    g_left_ua, g_right_ua, garment_contour = find_garment_underarms(garment_mask, gw)
    print(f"Garment underarms: L={g_left_ua}  R={g_right_ua}")

    print("\nBuilding AvatarRepresentation...")
    rep = build_avatar_representation()
    aw, ah = rep.image.size

    a_left_underarm_y, _ = find_semantic_underarm(rep.torso_mask, rep.upper_arms_mask, aw, "left")
    a_right_underarm_y, _ = find_semantic_underarm(rep.torso_mask, rep.upper_arms_mask, aw, "right")

    a_left_row = rep.torso_mask[a_left_underarm_y]
    a_left_xs = np.where(a_left_row[:aw//2])[0]
    a_left_ua = (int(a_left_xs.min()), a_left_underarm_y)

    a_right_row = rep.torso_mask[a_right_underarm_y]
    a_right_xs = np.where(a_right_row[aw//2:])[0] + aw // 2
    a_right_ua = (int(a_right_xs.max()), a_right_underarm_y)

    print(f"Avatar semantic underarms: L={a_left_ua}  R={a_right_ua}")

    # Garment's OWN labels are image-space (small-x = "left"); avatar's
    # are anatomical-space (per earlier finding, small-x = "right").
    # Pairing by physical x-position, not label name.
    garment_small_x = g_left_ua if g_left_ua[0] < g_right_ua[0] else g_right_ua
    garment_large_x = g_right_ua if g_left_ua[0] < g_right_ua[0] else g_left_ua
    avatar_small_x = a_left_ua if a_left_ua[0] < a_right_ua[0] else a_right_ua
    avatar_large_x = a_right_ua if a_left_ua[0] < a_right_ua[0] else a_left_ua

    garment_underarm_width = garment_large_x[0] - garment_small_x[0]
    avatar_underarm_width = avatar_large_x[0] - avatar_small_x[0]
    scale = avatar_underarm_width / garment_underarm_width

    print(f"\nGarment underarm width: {garment_underarm_width}px")
    print(f"Avatar underarm width:  {avatar_underarm_width}px")
    print(f"Scale factor: {scale:.4f}")

    # Transform: scale the garment uniformly, then translate so its
    # small-x underarm lands exactly on the avatar's small-x underarm.
    # Rotation is NOT applied — garment shape preserved exactly as-is.
    garment_arr = np.array(garment_img)
    new_w = int(gw * scale)
    new_h = int(gh * scale)
    resized_garment = garment_img.resize((new_w, new_h), Image.LANCZOS)

    scaled_underarm_x = garment_small_x[0] * scale
    scaled_underarm_y = garment_small_x[1] * scale

    offset_x = int(avatar_small_x[0] - scaled_underarm_x)
    offset_y = int(avatar_small_x[1] - scaled_underarm_y)

    # Composite onto a copy of the avatar image
    result = rep.image.convert("RGBA").copy()
    result.paste(resized_garment, (offset_x, offset_y), resized_garment)

    # Overlay: avatar torso outline + garment outline + both underarm pairs
    draw = ImageDraw.Draw(result)

    torso_contours, _ = cv2.findContours(
        rep.torso_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for c in torso_contours:
        pts = [(int(p[0][0]), int(p[0][1])) for p in c]
        if len(pts) > 2:
            draw.line(pts + [pts[0]], fill=(0, 255, 0), width=2)

    for pt, color, label in [
        (a_left_ua, (255, 0, 255), "A_L_UA"),
        (a_right_ua, (0, 200, 255), "A_R_UA"),
    ]:
        x, y = pt
        draw.ellipse([x-6, y-6, x+6, y+6], outline=color, width=3)
        draw.text((x+8, y-6), label, fill=color)

    result.convert("RGB").save(f"{OUTPUT_DIR}/correspondence_anchors_test.png")
    print(f"\nSaved: {OUTPUT_DIR}/correspondence_anchors_test.png")
    print(f"\n>>> DECISION GATE: aligned only at the underarms, does the "
          f"garment's own silhouette naturally sit around the avatar's "
          f"torso outline (green), or does it clearly diverge in width "
          f"or shape — revealing the real mismatch before any deformation "
          f"is attempted?")


if __name__ == "__main__":
    main()