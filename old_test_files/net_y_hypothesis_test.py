"""
Tests the |net_y_change| hypothesis against all six already-
confirmed sleeve/non-sleeve directions. Pure diagnostic — does NOT
modify classify_direction() or introduce any new production code.

Hypothesis: the sleeve direction is whichever of the two has the
SMALLER absolute net_y_change over the first 300px.

Run: python net_y_hypothesis_test.py
"""

import io
import requests
import numpy as np
import cv2
from PIL import Image

ALPHA_THRESHOLD = 10
MIN_DEFECT_DEPTH = 1000
MEASURE_WINDOW_PX = 300

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"

# Ground truth, independently confirmed via full trajectory tracing
# earlier tonight — not assumed, verified per-garment
TEST_CASES = [
    {"item_id": "ac7fffb7-2e3a-40f8-894a-0e85fc245373", "label": "polo",
     "confirmed_sleeve": {"LEFT": "BACKWARD", "RIGHT": "FORWARD"}},
    {"item_id": "25989827-db9d-4d65-90b3-12f5357b0b44", "label": "hanging_shirt",
     "confirmed_sleeve": {"LEFT": "BACKWARD", "RIGHT": "FORWARD"}},
    {"item_id": "2ea4bb57-a358-4349-8aaf-204ded0772ab", "label": "rugby",
     "confirmed_sleeve": {"LEFT": "FORWARD", "RIGHT": "BACKWARD"}},
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
    points_in_window = [(idx, d) for idx, d in ordered if d <= window_px]
    if len(points_in_window) < 2:
        return None
    first_idx, _ = points_in_window[0]
    last_idx, _ = points_in_window[-1]
    y_first = int(contour[first_idx][0][1])
    y_last = int(contour[last_idx][0][1])
    return y_last - y_first


def main():
    correct = 0
    total = 0
    margins = []

    for case in TEST_CASES:
        print(f"\n{'='*75}")
        print(f"{case['item_id']} — {case['label']}")
        print('='*75)

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

            net_y_fwd = get_net_y_change(main_contour, forward, MEASURE_WINDOW_PX)
            net_y_bwd = get_net_y_change(main_contour, backward, MEASURE_WINDOW_PX)

            abs_fwd = abs(net_y_fwd)
            abs_bwd = abs(net_y_bwd)

            predicted_sleeve = "FORWARD" if abs_fwd < abs_bwd else "BACKWARD"
            confirmed_sleeve = case["confirmed_sleeve"][side_name]
            agrees = predicted_sleeve == confirmed_sleeve
            margin = abs(abs_fwd - abs_bwd)

            total += 1
            if agrees:
                correct += 1
                margins.append(margin)

            print(f"\n  {side_name}:")
            print(f"    FORWARD  net_y_change={net_y_fwd:+4d}  |net_y|={abs_fwd}")
            print(f"    BACKWARD net_y_change={net_y_bwd:+4d}  |net_y|={abs_bwd}")
            print(f"    Rule predicts: {predicted_sleeve}   Confirmed sleeve: {confirmed_sleeve}   "
                  f"{'✅ MATCH' if agrees else '❌ MISMATCH'}   margin={margin}px")

    print(f"\n\n{'='*75}")
    print(f"FINAL RESULT: {correct}/{total} correct")
    if margins:
        print(f"Margins on correct predictions: {margins}")
        print(f"Smallest margin: {min(margins)}px   Largest margin: {max(margins)}px   "
              f"Mean margin: {np.mean(margins):.1f}px")
    print('='*75)


if __name__ == "__main__":
    main()