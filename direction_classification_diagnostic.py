"""
Direction-classification diagnostic — comparing candidate features
across confirmed-correct arcs (polo, hanging shirt) and the
confirmed-wrong arc (rugby), to find what genuinely distinguishes
sleeve direction from torso/collar direction.

Measures, for BOTH directions from each underarm, over the first
~300px:
  - sustained lateral displacement (mean |x - underarm_x|, not just max)
  - distance from garment's vertical midline
  - monotonicity of x movement (fraction of steps moving same way)
  - net y-direction (does it move toward hem or toward collar?)
  - distance from torso/neck region (using known collar_top_percent 
    reference where available)
  - eventual extremum/turning behavior (does x reverse at all?)

Does NOT select a winner. Pure measurement, for comparison.

Run: python direction_classification_diagnostic.py
"""

import io
import requests
import numpy as np
import cv2
from PIL import Image

ALPHA_THRESHOLD = 10
MIN_DEFECT_DEPTH = 1000
MEASURE_WINDOW_PX = 300

TEST_CASES = [
    {"item_id": "ac7fffb7-2e3a-40f8-894a-0e85fc245373", "label": "polo (CORRECT arc known)"},
    {"item_id": "25989827-db9d-4d65-90b3-12f5357b0b44", "label": "hanging_shirt (CORRECT arc known)"},
    {"item_id": "2ea4bb57-a358-4349-8aaf-204ded0772ab", "label": "rugby (WRONG arc selected by v7)"},
]

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"


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


def measure_arc_features(contour, arc_dict, underarm_x, underarm_y, midline_x, window_px):
    ordered = sorted(arc_dict.items(), key=lambda kv: kv[1])
    points = [(int(contour[idx][0][0]), int(contour[idx][0][1]), d)
              for idx, d in ordered if d <= window_px]

    if len(points) < 3:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    mean_lateral_disp = np.mean([abs(x - underarm_x) for x in xs])
    max_lateral_disp = max(abs(x - underarm_x) for x in xs)
    mean_dist_from_midline = np.mean([abs(x - midline_x) for x in xs])

    x_diffs = np.diff(xs)
    moving_positive = np.sum(x_diffs > 0)
    moving_negative = np.sum(x_diffs < 0)
    monotonicity = max(moving_positive, moving_negative) / max(len(x_diffs), 1)

    net_y_change = ys[-1] - ys[0]  # positive = moved DOWN (toward hem), negative = UP (toward collar)

    min_x, max_x = min(xs), max(xs)
    initial_x = xs[0]
    reaches_extremum_then_returns = False
    if initial_x == max(xs[:5]) or initial_x == min(xs[:5]):
        far_idx = xs.index(max_x) if abs(max_x - underarm_x) > abs(min_x - underarm_x) else xs.index(min_x)
        if 0 < far_idx < len(xs) - 1:
            reaches_extremum_then_returns = True

    return {
        "mean_lateral_disp": round(mean_lateral_disp, 1),
        "max_lateral_disp": round(max_lateral_disp, 1),
        "mean_dist_from_midline": round(mean_dist_from_midline, 1),
        "monotonicity": round(monotonicity, 2),
        "net_y_change": net_y_change,
        "reaches_extremum_then_returns": reaches_extremum_then_returns,
    }


def main():
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
        midline_x = w_img / 2

        left_idx, right_idx = find_underarm_indices(main_contour, w_img)

        for side_name, underarm_idx in [("LEFT", left_idx), ("RIGHT", right_idx)]:
            underarm_x = int(main_contour[underarm_idx][0][0])
            underarm_y = int(main_contour[underarm_idx][0][1])
            forward, backward = contour_arc_distances(main_contour, underarm_idx)

            print(f"\n  {side_name} underarm=({underarm_x},{underarm_y})")

            for dir_name, arc_dict in [("FORWARD", forward), ("BACKWARD", backward)]:
                features = measure_arc_features(main_contour, arc_dict, underarm_x, underarm_y, midline_x, MEASURE_WINDOW_PX)
                if features is None:
                    continue
                print(f"    {dir_name}: {features}")


if __name__ == "__main__":
    main()