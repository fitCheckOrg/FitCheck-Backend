"""
Tests whether x-RANGE vs y-RANGE (max-min, not net change) over the
sleeve arc correctly predicts which boundary/extremum type each
garment needs — BEFORE building the full layered pipeline.

Uses the already-validated Layer 1 rule to get the correct sleeve
arc for each garment, then measures range in both dimensions.

Known ground truth (from tonight's investigation):
  polo          → needs X-extremum (extended sleeve)
  hanging_shirt → needs Y-extremum (hanging sleeve)
  rugby         → needs Y-extremum (hanging sleeve)
  dress_shirt   → needs Y-extremum (hanging sleeve)

Run: python orientation_signal_test.py
"""

import io
import requests
import numpy as np
import cv2
from PIL import Image

ALPHA_THRESHOLD = 10
MIN_DEFECT_DEPTH = 1000
NET_Y_WINDOW_PX = 300
RANGE_WINDOW_PX = 300

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"

TEST_CASES = [
    {"item_id": "ac7fffb7-2e3a-40f8-894a-0e85fc245373", "label": "polo", "expected": "X"},
    {"item_id": "25989827-db9d-4d65-90b3-12f5357b0b44", "label": "hanging_shirt", "expected": "Y"},
    {"item_id": "2ea4bb57-a358-4349-8aaf-204ded0772ab", "label": "rugby", "expected": "Y"},
    {"item_id": "6a53c00f-4f84-4e64-ad3b-5d2753504481", "label": "dress_shirt", "expected": "Y"},
]


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255


def find_underarm_indices(main_contour, w_img):
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
    return left_candidates[0][1], right_candidates[0][1]


def contour_arc_distances(contour, start_idx):
    n = len(contour)
    forward, backward = {}, {}
    dist = 0.0
    for step in range(n):
        idx = (start_idx + step) % n
        forward[idx] = dist
        next_idx = (idx + 1) % n
        dist += np.linalg.norm(contour[next_idx][0].astype(float) - contour[idx][0].astype(float))
    dist = 0.0
    for step in range(n):
        idx = (start_idx - step) % n
        backward[idx] = dist
        prev_idx = (idx - 1) % n
        dist += np.linalg.norm(contour[prev_idx][0].astype(float) - contour[idx][0].astype(float))
    return forward, backward


def get_net_y_change(contour, arc_dict, window_px):
    ordered = sorted(arc_dict.items(), key=lambda kv: kv[1])
    points = [(idx, d) for idx, d in ordered if d <= window_px]
    if len(points) < 2:
        return None
    y_first = int(contour[points[0][0]][0][1])
    y_last = int(contour[points[-1][0]][0][1])
    return y_last - y_first


def get_range(contour, arc_dict, window_px):
    ordered = sorted(arc_dict.items(), key=lambda kv: kv[1])
    points = [(idx, d) for idx, d in ordered if d <= window_px]
    xs = [int(contour[idx][0][0]) for idx, d in points]
    ys = [int(contour[idx][0][1]) for idx, d in points]
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    return x_range, y_range


def main():
    correct = 0
    total = 0

    for case in TEST_CASES:
        print(f"\n{'='*70}")
        print(f"{case['item_id']} — {case['label']}  (expected: {case['expected']}-extremum)")
        print('='*70)

        result_data = requests.get(
            f"http://127.0.0.1:8000/api/wardrobe/{case['item_id']}?user_id={AVATAR_USER_ID}"
        ).json()["data"]
        image_bytes = requests.get(result_data["clean_image_url"]).content

        binary_mask = get_binary_mask(image_bytes)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_contour = max(contours, key=cv2.contourArea)
        w_img = binary_mask.shape[1]

        left_idx, right_idx = find_underarm_indices(main_contour, w_img)

        for side_name, underarm_idx in [("LEFT", left_idx), ("RIGHT", right_idx)]:
            forward, backward = contour_arc_distances(main_contour, underarm_idx)

            net_y_fwd = get_net_y_change(main_contour, forward, NET_Y_WINDOW_PX)
            net_y_bwd = get_net_y_change(main_contour, backward, NET_Y_WINDOW_PX)
            sleeve_arc = forward if abs(net_y_fwd) < abs(net_y_bwd) else backward

            x_range, y_range = get_range(main_contour, sleeve_arc, RANGE_WINDOW_PX)
            predicted = "X" if x_range > y_range else "Y"
            agrees = predicted == case["expected"]

            total += 1
            if agrees:
                correct += 1

            print(f"  {side_name}: x_range={x_range}  y_range={y_range}  "
                  f"predicted={predicted}-extremum  {'✅' if agrees else '❌'}")

    print(f"\n\n{'='*70}")
    print(f"ORIENTATION SIGNAL RESULT: {correct}/{total}")
    print('='*70)


if __name__ == "__main__":
    main()