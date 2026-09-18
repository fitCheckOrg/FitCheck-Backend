"""
Characterizes the adjacent-peak phenomenon across all 8 cases.
Pure diagnostic, per explicit instruction: no MIN_PEAK_SEPARATION
change, no merging logic implemented yet.

Answers three questions:
1. How many "adjacent strong peaks" occur near confirmed cuffs?
2. What's the arc-distance distribution between them?
3. Would a naive merge accidentally combine genuinely distinct
   features elsewhere in the dataset? (checked by also reporting
   arc-gaps between ALL top candidates, not just near-cuff ones)

Run: python adjacent_peak_characterization.py
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
MIN_ARC_BEFORE_PEAK_SEARCH = 10
MIN_PEAK_SEPARATION = 10  # unchanged, current fixture value
TOP_K = 8
NEAR_CUFF_RADIUS_PX = 100  # for identifying "peaks near the confirmed cuff"
STRONG_PEAK_RATIO = 0.85   # a peak is "comparably strong" if within 
                             # this fraction of the strongest nearby peak

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"

TEST_CASES = [
    {"item_id": "ac7fffb7-2e3a-40f8-894a-0e85fc245373", "label": "polo",
     "confirmed_cuff": {"LEFT": (31, 154), "RIGHT": (439, 142)}},
    {"item_id": "25989827-db9d-4d65-90b3-12f5357b0b44", "label": "hanging_shirt",
     "confirmed_cuff": {"LEFT": (26, 457), "RIGHT": (442, 447)}},
    {"item_id": "2ea4bb57-a358-4349-8aaf-204ded0772ab", "label": "rugby",
     "confirmed_cuff": {"LEFT": (51, 488), "RIGHT": (430, 459)}},
    {"item_id": "6a53c00f-4f84-4e64-ad3b-5d2753504481", "label": "dress_shirt",
     "confirmed_cuff": {"LEFT": (54, 633), "RIGHT": (166, 660)}},
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


def find_top_k_peaks(ordered_arc, curvatures, top_k, min_separation=MIN_PEAK_SEPARATION):
    candidates = []
    for i in range(1, len(ordered_arc) - 1):
        idx, d = ordered_arc[i]
        if d < MIN_ARC_BEFORE_PEAK_SEARCH:
            continue
        prev_idx, _ = ordered_arc[i - 1]
        next_idx, _ = ordered_arc[i + 1]
        c = curvatures.get(idx, 0.0)
        if c > curvatures.get(prev_idx, 0.0) and c > curvatures.get(next_idx, 0.0):
            candidates.append((c, d, idx))
    candidates.sort(reverse=True)
    selected = []
    for c, d, idx in candidates:
        if not any(abs(d - sd) < min_separation for _, sd, _ in selected):
            selected.append((c, d, idx))
        if len(selected) >= top_k:
            break
    return selected


def main():
    all_near_cuff_gaps = []
    all_gaps_between_top_candidates = []

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
            net_y_fwd = get_net_y_change(main_contour, forward, NET_Y_WINDOW_PX)
            net_y_bwd = get_net_y_change(main_contour, backward, NET_Y_WINDOW_PX)
            sleeve_arc = forward if abs(net_y_fwd) < abs(net_y_bwd) else backward

            ordered = sorted(sleeve_arc.items(), key=lambda kv: kv[1])
            curvatures = compute_curvature(main_contour, ordered)
            peaks = find_top_k_peaks(ordered, curvatures, TOP_K)

            confirmed = case["confirmed_cuff"][side_name]

            # Find all peaks physically near the confirmed cuff (Euclidean)
            near_cuff_peaks = []
            for c, d, idx in peaks:
                pt = tuple(int(v) for v in main_contour[idx][0])
                pixel_dist = np.hypot(pt[0]-confirmed[0], pt[1]-confirmed[1])
                if pixel_dist < NEAR_CUFF_RADIUS_PX:
                    near_cuff_peaks.append((c, d, pt, pixel_dist))

            print(f"\n  {side_name} — {len(near_cuff_peaks)} peak(s) within "
                  f"{NEAR_CUFF_RADIUS_PX}px of confirmed cuff {confirmed}:")
            for c, d, pt, pixel_dist in sorted(near_cuff_peaks, key=lambda x: -x[0]):
                print(f"    curvature={c:6.1f}°  arc={d:7.1f}  pos={pt}  "
                      f"pixel_dist_from_confirmed={pixel_dist:.1f}px")

            # Q1 & Q2: arc-gaps between comparably-strong near-cuff peaks
            if len(near_cuff_peaks) >= 2:
                near_cuff_peaks.sort(key=lambda x: -x[0])
                strongest_c = near_cuff_peaks[0][0]
                for i in range(1, len(near_cuff_peaks)):
                    c, d, pt, _ = near_cuff_peaks[i]
                    if c >= strongest_c * STRONG_PEAK_RATIO:
                        arc_gap = abs(d - near_cuff_peaks[0][1])
                        print(f"    >>> Comparably strong pair: arc_gap={arc_gap:.1f}px "
                              f"(curvatures {strongest_c:.1f}° vs {c:.1f}°)")
                        all_near_cuff_gaps.append(arc_gap)

            # Q3: arc-gaps between ALL consecutive top candidates (not just near-cuff)
            # — to check whether a merge threshold would risk combining unrelated features
            peaks_by_arc = sorted(peaks, key=lambda p: p[1])
            for i in range(len(peaks_by_arc) - 1):
                d1 = peaks_by_arc[i][1]
                d2 = peaks_by_arc[i+1][1]
                gap = d2 - d1
                if gap < 200:  # only report genuinely nearby pairs, not distant ones
                    all_gaps_between_top_candidates.append(gap)

    print(f"\n\n{'='*75}")
    print("SUMMARY")
    print('='*75)
    print(f"\nQ1/Q2 — Comparably-strong near-cuff peak pairs found: {len(all_near_cuff_gaps)}")
    if all_near_cuff_gaps:
        print(f"  Arc-gap distribution: {sorted(all_near_cuff_gaps)}")
        print(f"  Min={min(all_near_cuff_gaps):.1f}  Max={max(all_near_cuff_gaps):.1f}  "
              f"Mean={np.mean(all_near_cuff_gaps):.1f}")

    print(f"\nQ3 — All nearby (<200px) consecutive top-candidate arc-gaps, "
          f"across the whole dataset: {len(all_gaps_between_top_candidates)}")
    if all_gaps_between_top_candidates:
        print(f"  Distribution: {sorted(all_gaps_between_top_candidates)}")
        print(f"  → Compare this against Q1/Q2's gaps: if they overlap significantly, "
              f"a single merge-distance threshold risks conflating genuinely distinct "
              f"features with true adjacent-duplicate detections.")


if __name__ == "__main__":
    main()