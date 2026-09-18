"""
Layer 2 diagnostic — characterizes WHY x never returns for the
dress shirt, using the NEWLY VALIDATED Layer 1 rule (min |net_y_
change|) to first confirm we're even looking at the correct sleeve
arc — this garment was never part of the 6/6 validation set.

Then traces the FULL x and y trajectory over an extended window,
explicitly counting every crossing of the underarm's x-coordinate
(not just the first), and checking whether Y offers an alternative,
more robust turning-point signal.

Pure diagnostic. No fallback proposed yet.

Run: python dress_shirt_geometric_characterization.py
"""

import io
import requests
import numpy as np
import cv2
from PIL import Image

ALPHA_THRESHOLD = 10
MIN_DEFECT_DEPTH = 1000
NET_Y_WINDOW_PX = 300
TRACE_CAP = 800
PRINT_INTERVAL = 20

ITEM_ID = "6a53c00f-4f84-4e64-ad3b-5d2753504481"
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
        underarm_x = int(main_contour[underarm_idx][0][0])
        underarm_pt = tuple(int(v) for v in main_contour[underarm_idx][0])
        forward, backward = contour_arc_distances(main_contour, underarm_idx)

        # Apply the NEW, validated Layer 1 rule for the first time on this garment
        net_y_fwd = get_net_y_change(main_contour, forward, NET_Y_WINDOW_PX)
        net_y_bwd = get_net_y_change(main_contour, backward, NET_Y_WINDOW_PX)
        sleeve_direction = "FORWARD" if abs(net_y_fwd) < abs(net_y_bwd) else "BACKWARD"
        sleeve_arc = forward if sleeve_direction == "FORWARD" else backward

        print(f"\n{'='*70}")
        print(f"{side_name} — underarm={underarm_pt}")
        print(f"  net_y FORWARD={net_y_fwd:+d} (|{abs(net_y_fwd)}|)   "
              f"net_y BACKWARD={net_y_bwd:+d} (|{abs(net_y_bwd)}|)")
        print(f"  Layer 1 selects: {sleeve_direction}")
        print(f"{'='*70}")

        ordered = sorted(sleeve_arc.items(), key=lambda kv: kv[1])

        crossing_count = 0
        last_side = None  # track which side of underarm_x we're on
        last_printed = -PRINT_INTERVAL
        max_y_seen, max_y_arc = -1, None
        min_y_seen, min_y_arc = 99999, None

        for idx, d in ordered:
            if d > TRACE_CAP:
                break
            x, y = main_contour[idx][0]
            x, y = int(x), int(y)

            current_side = "above" if x > underarm_x else ("below" if x < underarm_x else "same")
            if last_side is not None and current_side != "same" and last_side != "same" and current_side != last_side:
                crossing_count += 1
                if crossing_count <= 10:
                    print(f"    >>> X-CROSSING #{crossing_count} at arc={d:.1f}px pos=({x},{y})")
            if current_side != "same":
                last_side = current_side

            if y > max_y_seen:
                max_y_seen, max_y_arc = y, d
            if y < min_y_seen:
                min_y_seen, min_y_arc = y, d

            if d - last_printed >= PRINT_INTERVAL:
                print(f"  arc={d:6.1f}px  pos=({x:3d},{y:3d})")
                last_printed = d

        print(f"\n  Total x-crossings of underarm_x within {TRACE_CAP}px: {crossing_count}")
        print(f"  Max y (lowest point) reached: {max_y_seen} at arc={max_y_arc:.1f}px")
        print(f"  Min y (highest point) reached: {min_y_seen} at arc={min_y_arc:.1f}px")


if __name__ == "__main__":
    main()