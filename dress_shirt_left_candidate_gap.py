"""
Investigates why dress-shirt LEFT's confirmed cuff (arc≈18.7) is
missing from the top-4 candidate list. Per the checkpoint: this
must be understood BEFORE touching the selector, since it may be a
candidate-generation/filtering issue, not a selection issue.

Prints the raw curvature profile from arc=0 (no MIN_ARC filter)
to see exactly what's happening in the excluded zone.

Run: python dress_shirt_left_candidate_gap.py
"""

import io
import requests
import numpy as np
import cv2
from PIL import Image

ALPHA_THRESHOLD = 10
MIN_DEFECT_DEPTH = 1000
CURVATURE_WINDOW = 8
NET_Y_WINDOW_PX = 300
MIN_ARC_BEFORE_PEAK_SEARCH = 30  # the value in question

ITEM_ID = "6a53c00f-4f84-4e64-ad3b-5d2753504481"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
CONFIRMED_CUFF = (54, 633)


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


def compute_curvature(contour, ordered_arc, window=CURVATURE_WINDOW):
    indices = [idx for idx, d in ordered_arc]
    pts = np.array([contour[i][0] for i in indices], dtype=float)
    n = len(pts)
    curvatures = {}
    for i in range(n):
        prev_i = max(0, i - window)
        next_i = min(n - 1, i + window)
        if next_i - prev_i < 2:
            curvatures[indices[i]] = 0.0
            continue
        v1 = pts[i] - pts[prev_i]
        v2 = pts[next_i] - pts[i]
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        curvatures[indices[i]] = 0.0 if norm1 == 0 or norm2 == 0 else \
            np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)))
    return curvatures


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
    underarm_pt = tuple(int(v) for v in main_contour[left_idx][0])

    forward, backward = contour_arc_distances(main_contour, left_idx)
    net_y_fwd = get_net_y_change(main_contour, forward, NET_Y_WINDOW_PX)
    net_y_bwd = get_net_y_change(main_contour, backward, NET_Y_WINDOW_PX)
    sleeve_arc = forward if abs(net_y_fwd) < abs(net_y_bwd) else backward
    direction = "FORWARD" if sleeve_arc is forward else "BACKWARD"

    ordered = sorted(sleeve_arc.items(), key=lambda kv: kv[1])
    curvatures = compute_curvature(main_contour, ordered)

    print(f"underarm={underarm_pt}  direction={direction}")
    print(f"confirmed_cuff={CONFIRMED_CUFF}")
    print(f"MIN_ARC_BEFORE_PEAK_SEARCH={MIN_ARC_BEFORE_PEAK_SEARCH}\n")

    print("Raw curvature profile from arc=0 to arc=60 (no filtering applied):")
    print(f"{'arc':>8} {'idx':>6} {'pos':>14} {'curvature':>10}  is_local_max?")
    for i, (idx, d) in enumerate(ordered):
        if d > 60:
            break
        x, y = main_contour[idx][0]
        c = curvatures.get(idx, 0.0)
        is_max = ""
        if 0 < i < len(ordered) - 1:
            prev_idx = ordered[i-1][0]
            next_idx = ordered[i+1][0]
            if c > curvatures.get(prev_idx, 0.0) and c > curvatures.get(next_idx, 0.0):
                is_max = "  <-- LOCAL MAX"
        excluded = "  [EXCLUDED by MIN_ARC filter]" if d < MIN_ARC_BEFORE_PEAK_SEARCH else ""
        print(f"{d:8.1f} {idx:6d} ({int(x):4d},{int(y):4d})   {c:8.1f}°{is_max}{excluded}")

    # Find exact peak curvature nearest confirmed cuff
    best_idx, best_dist = None, float('inf')
    for idx, d in ordered:
        x, y = main_contour[idx][0]
        dist = np.hypot(int(x) - CONFIRMED_CUFF[0], int(y) - CONFIRMED_CUFF[1])
        if dist < best_dist:
            best_dist, best_idx = dist, idx
    best_d = sleeve_arc[best_idx]
    print(f"\nClosest contour point to confirmed cuff: "
          f"arc={best_d:.1f}  curvature={curvatures.get(best_idx, 0.0):.1f}°  "
          f"distance_from_confirmed={best_dist:.1f}px")


if __name__ == "__main__":
    main()