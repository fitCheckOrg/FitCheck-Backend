"""
Cuff detection investigation — standalone, prior to any sleeve
fitting attempt.

Hypothesis: within a sleeve's known column range (already 
established via GPT horizontal partition), the cuff point is the 
contour point with MAXIMUM distance from that side's underarm 
point. Orientation-agnostic by design — works whether the sleeve 
hangs down or extends outward, without needing the abandoned 
categorical "orientation" label.

Tests against two known cases: the hanging-sleeve shirt (where 
B1/B2 curvature peaks were already visually confirmed as cuffs 
in an earlier investigation) and the polo (clean extended-sleeve 
baseline).

Does NOT touch torso/hem code, GarmentLandmarkSet, or any fitting 
math. Pure geometry investigation.

Run: python cuff_detection_test.py
"""

import io
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw

ALPHA_THRESHOLD = 10
MIN_DEFECT_DEPTH = 1000

TEST_CASES = [
    {"item_id": "25989827-db9d-4d65-90b3-12f5357b0b44", "label": "hanging_sleeve_shirt"},
    {"item_id": "ac7fffb7-2e3a-40f8-894a-0e85fc245373", "label": "polo_baseline"},
]

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255, image


def find_underarms(main_contour, w_img):
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
    left_underarm = tuple(int(v) for v in main_contour[left_candidates[0][1]][0])
    right_underarm = tuple(int(v) for v in main_contour[right_candidates[0][1]][0])
    return left_underarm, right_underarm


def find_cuff_in_column_range(main_contour, underarm_point, x_min, x_max):
    """
    THE HYPOTHESIS: cuff = contour point within [x_min, x_max]
    with maximum Euclidean distance from underarm_point.
    """
    ux, uy = underarm_point
    best_point = None
    best_dist = -1

    for pt in main_contour[:, 0, :]:
        x, y = int(pt[0]), int(pt[1])
        if x_min <= x <= x_max:
            dist = np.hypot(x - ux, y - uy)
            if dist > best_dist:
                best_dist = dist
                best_point = (x, y)

    return best_point, best_dist


def main():
    for case in TEST_CASES:
        print(f"\n{'='*70}")
        print(f"{case['item_id']} — {case['label']}")
        print('='*70)

        result_data = requests.get(
            f"http://127.0.0.1:8000/api/wardrobe/{case['item_id']}?user_id={AVATAR_USER_ID}"
        ).json()["data"]
        image_bytes = requests.get(result_data["clean_image_url"]).content

        binary_mask, original = get_binary_mask(image_bytes)
        h_img, w_img = binary_mask.shape
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_contour = max(contours, key=cv2.contourArea)

        left_ua, right_ua = find_underarms(main_contour, w_img)
        print(f"left_underarm: {left_ua}  right_underarm: {right_ua}")

        # Sleeve column ranges — same convention used throughout 
        # tonight (0-20% left sleeve, 80-100% right sleeve, matching 
        # GPT's horizontal partition pattern)
        left_sleeve_range = (0, int(w_img * 0.20))
        right_sleeve_range = (int(w_img * 0.80), w_img)

        left_cuff, left_dist = find_cuff_in_column_range(main_contour, left_ua, *left_sleeve_range)
        right_cuff, right_dist = find_cuff_in_column_range(main_contour, right_ua, *right_sleeve_range)

        print(f"\nleft_cuff candidate: {left_cuff}  distance_from_underarm={left_dist:.1f}px  "
              f"pos=({left_cuff[0]/w_img*100:.1f}%, {left_cuff[1]/h_img*100:.1f}%)")
        print(f"right_cuff candidate: {right_cuff}  distance_from_underarm={right_dist:.1f}px  "
              f"pos=({right_cuff[0]/w_img*100:.1f}%, {right_cuff[1]/h_img*100:.1f}%)")

        canvas = original.convert("RGB").copy()
        draw = ImageDraw.Draw(canvas)
        pts = [(int(p[0][0]), int(p[0][1])) for p in main_contour]
        draw.line(pts + [pts[0]], fill=(80, 80, 80), width=1)

        for pt, color, label in [(left_ua, (0, 255, 0), "L_underarm"),
                                    (right_ua, (0, 255, 0), "R_underarm"),
                                    (left_cuff, (255, 0, 0), "L_cuff"),
                                    (right_cuff, (0, 200, 255), "R_cuff")]:
            x, y = pt
            draw.ellipse([x-10, y-10, x+10, y+10], outline=color, width=3)
            draw.text((x+12, y), label, fill=color)

        filename = f"cuff_test_{case['label']}.png"
        canvas.save(filename)
        print(f"Saved: {filename}")


if __name__ == "__main__":
    main()