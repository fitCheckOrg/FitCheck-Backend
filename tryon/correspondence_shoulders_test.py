"""
Adds shoulder/collar correspondence on top of the already-validated
underarm alignment. Uses the SAME scale+translation derived from
underarms only — does NOT recompute a new transform. Then checks
where the garment's own shoulder point lands relative to the
avatar's real MediaPipe shoulder, under that existing transform.

This directly tests: does the underarm-only transform also satisfy
shoulder correspondence, or does it reveal the need for more than
a similarity transform?

Run: python -m tryon.correspondence_shoulders_test
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
COLLAR_EXCLUSION_MARGIN = 0.15  # fraction of width excluded around center, avoiding the collar peak
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
    return left, right


def find_garment_shoulder_points(mask, w_img):
    """
    Topmost mask pixel on each outer flank, excluding a central
    margin around the collar peak. Gives the collar/shoulder-seam
    junction — the same feature visually identified earlier tonight.
    """
    margin_px = int(w_img * COLLAR_EXCLUSION_MARGIN)
    mid = w_img // 2

    left_region = mask[:, :mid - margin_px]
    right_region = mask[:, mid + margin_px:]

    left_ys, left_xs = np.where(left_region)
    right_ys, right_xs = np.where(right_region)

    left_top_y = left_ys.min()
    left_top_x = left_xs[left_ys == left_top_y].mean()

    right_top_y = right_ys.min()
    right_top_x = right_xs[right_ys == right_top_y].mean() + (mid + margin_px)

    return (int(left_top_x), int(left_top_y)), (int(right_top_x), int(right_top_y))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Fetching golden garment...")
    garment_resp = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{GOLDEN_GARMENT_ID}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    garment_bytes = requests.get(garment_resp["clean_image_url"]).content
    garment_mask, garment_img = get_binary_mask(garment_bytes)
    gw, gh = garment_img.size

    g_left_ua, g_right_ua = find_garment_underarms(garment_mask, gw)
    g_left_shoulder, g_right_shoulder = find_garment_shoulder_points(garment_mask, gw)
    print(f"Garment underarms: L={g_left_ua}  R={g_right_ua}")
    print(f"Garment shoulder points: L={g_left_shoulder}  R={g_right_shoulder}")

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
    print(f"Avatar MediaPipe shoulders: L={rep.left_shoulder}  R={rep.right_shoulder}")

    # SAME transform as correspondence_anchors_test.py — underarms only.
    garment_small_x = g_left_ua if g_left_ua[0] < g_right_ua[0] else g_right_ua
    garment_large_x = g_right_ua if g_left_ua[0] < g_right_ua[0] else g_left_ua
    avatar_small_x = a_left_ua if a_left_ua[0] < a_right_ua[0] else a_right_ua
    avatar_large_x = a_right_ua if a_left_ua[0] < a_right_ua[0] else a_left_ua

    garment_underarm_width = garment_large_x[0] - garment_small_x[0]
    avatar_underarm_width = avatar_large_x[0] - avatar_small_x[0]
    scale = avatar_underarm_width / garment_underarm_width

    scaled_underarm_x = garment_small_x[0] * scale
    scaled_underarm_y = garment_small_x[1] * scale
    offset_x = avatar_small_x[0] - scaled_underarm_x
    offset_y = avatar_small_x[1] - scaled_underarm_y

    def transform_point(pt):
        return (pt[0] * scale + offset_x, pt[1] * scale + offset_y)

    # Apply the EXISTING transform to the garment's shoulder points —
    # not recomputing anything, just checking where they land
    g_small_shoulder = g_left_shoulder if g_left_shoulder[0] < g_right_shoulder[0] else g_right_shoulder
    g_large_shoulder = g_right_shoulder if g_left_shoulder[0] < g_right_shoulder[0] else g_left_shoulder

    transformed_small_shoulder = transform_point(g_small_shoulder)
    transformed_large_shoulder = transform_point(g_large_shoulder)

    a_small_shoulder = rep.left_shoulder if rep.left_shoulder[0] < rep.right_shoulder[0] else rep.right_shoulder
    a_large_shoulder = rep.right_shoulder if rep.left_shoulder[0] < rep.right_shoulder[0] else rep.left_shoulder

    dist_small = np.hypot(transformed_small_shoulder[0] - a_small_shoulder[0],
                            transformed_small_shoulder[1] - a_small_shoulder[1])
    dist_large = np.hypot(transformed_large_shoulder[0] - a_large_shoulder[0],
                            transformed_large_shoulder[1] - a_large_shoulder[1])

    print(f"\nGarment shoulder (transformed) vs Avatar MediaPipe shoulder:")
    print(f"  Small-x side: transformed={transformed_small_shoulder}  avatar={a_small_shoulder}  dist={dist_small:.1f}px")
    print(f"  Large-x side: transformed={transformed_large_shoulder}  avatar={a_large_shoulder}  dist={dist_large:.1f}px")

    # Visualization
    new_w, new_h = int(gw * scale), int(gh * scale)
    resized_garment = garment_img.resize((new_w, new_h), Image.LANCZOS)
    result = rep.image.convert("RGBA").copy()
    result.paste(resized_garment, (int(offset_x), int(offset_y)), resized_garment)

    draw = ImageDraw.Draw(result)
    for pt, color, label in [
        (transformed_small_shoulder, (255, 0, 255), "garment_shoulder(transformed)"),
        (transformed_large_shoulder, (255, 0, 255), "garment_shoulder(transformed)"),
        (a_small_shoulder, (0, 255, 0), "avatar_shoulder(MediaPipe)"),
        (a_large_shoulder, (0, 255, 0), "avatar_shoulder(MediaPipe)"),
    ]:
        x, y = pt
        draw.ellipse([x-6, y-6, x+6, y+6], outline=color, width=3)

    result.convert("RGB").save(f"{OUTPUT_DIR}/correspondence_shoulders_test.png")
    print(f"\nSaved: {OUTPUT_DIR}/correspondence_shoulders_test.png")
    print(f"\n>>> DECISION GATE: are the distances small enough that the "
          f"underarm-only transform ALSO satisfies shoulder correspondence "
          f"(similarity transform sufficient), or is the gap large enough "
          f"to require additional degrees of freedom (affine/piecewise)?")


if __name__ == "__main__":
    main()