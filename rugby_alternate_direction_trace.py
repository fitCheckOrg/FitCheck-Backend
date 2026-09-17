"""
Full trajectory trace for rugby's UNCONFIRMED direction — the one
v7's classify_direction() did NOT select. We've only ever verified
where the WRONG direction goes (collar); this checks whether the
other direction actually reaches a real cuff, rather than assuming
it does from encouraging feature numbers alone.

Run: python rugby_alternate_direction_trace.py
"""

import io
import requests
import numpy as np
import cv2
from PIL import Image

ALPHA_THRESHOLD = 10
MIN_DEFECT_DEPTH = 1000
PRINT_INTERVAL = 15
TRACE_CAP = 500

ITEM_ID = "2ea4bb57-a358-4349-8aaf-204ded0772ab"
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


def print_trace(contour, arc_dict, label):
    ordered = sorted(arc_dict.items(), key=lambda kv: kv[1])
    print(f"\n  --- {label} ---")
    last_printed = -PRINT_INTERVAL
    max_y_seen = -1
    max_y_arc = None
    for idx, d in ordered:
        if d > TRACE_CAP:
            break
        x, y = contour[idx][0]
        if y > max_y_seen:
            max_y_seen = y
            max_y_arc = d
        if d - last_printed >= PRINT_INTERVAL:
            print(f"    arc={d:6.1f}px  pos=({int(x):3d},{int(y):3d})")
            last_printed = d
    print(f"    Deepest y reached: {max_y_seen} at arc={max_y_arc:.1f}px")


def main():
    result_data = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{ITEM_ID}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    image_bytes = requests.get(result_data["clean_image_url"]).content

    binary_mask = get_binary_mask(image_bytes)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)
    w_img = binary_mask.shape[1]

    left_idx, right_idx = find_underarm_indices(main_contour, w_img)

    for side_name, underarm_idx in [("LEFT", left_idx), ("RIGHT", right_idx)]:
        underarm_pt = tuple(int(v) for v in main_contour[underarm_idx][0])
        forward, backward = contour_arc_distances(main_contour, underarm_idx)

        print(f"\n{'='*60}")
        print(f"{side_name} — underarm={underarm_pt}")
        print(f"{'='*60}")
        print_trace(main_contour, forward, "FORWARD")
        print_trace(main_contour, backward, "BACKWARD")


if __name__ == "__main__":
    main()