"""
Tests whether TANGENT DIRECTION REVERSAL (not literal sign match)
in both dx and dy separates the true cuff from competing curvature
peaks — across all 8 known sleeve cases, fully unbounded.

No boundary logic. No distance constants. Layer 1 (validated
direction) is the only prior stage used. Top 3 real local maxima,
properly separated (not raw top-3 values, which can be adjacent
samples of the same physical corner).

For each candidate: compare tangent direction (sign of dx, sign of
dy) at a point clearly BEFORE it to a point clearly AFTER it, on
the arc. A reversal is sign(before) != sign(after) — orientation-
agnostic by construction.

Run: python cuff_candidate_signature_test.py
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
MIN_ARC_BEFORE_PEAK_SEARCH = 30
MIN_PEAK_SEPARATION = 10
TANGENT_CHECK_OFFSET_PX = 20  # how far before/after the peak to sample tangent direction

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


def compute_curvature_and_deltas(contour, ordered_arc, window=CURVATURE_WINDOW):
    indices = [idx for idx, d in ordered_arc]
    pts = np.array([contour[i][0] for i in indices], dtype=float)
    n = len(pts)
    results = {}
    for i in range(n):
        prev_i = max(0, i - window)
        next_i = min(n - 1, i + window)
        if next_i - prev_i < 2:
            results[indices[i]] = (0.0, 0.0, 0.0)
            continue
        v1 = pts[i] - pts[prev_i]
        v2 = pts[next_i] - pts[i]
        dx = pts[next_i][0] - pts[prev_i][0]
        dy = pts[next_i][1] - pts[prev_i][1]
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        curvature = 0.0 if norm1 == 0 or norm2 == 0 else \
            np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)))
        results[indices[i]] = (dx, dy, curvature)
    return results


def find_top_local_peaks(ordered_arc, deltas, top_k=3, min_separation=MIN_PEAK_SEPARATION):
    candidates = []
    for i in range(1, len(ordered_arc) - 1):
        idx, d = ordered_arc[i]
        if d < MIN_ARC_BEFORE_PEAK_SEARCH:
            continue
        prev_idx, _ = ordered_arc[i - 1]
        next_idx, _ = ordered_arc[i + 1]
        c = deltas[idx][2]
        if c > deltas[prev_idx][2] and c > deltas[next_idx][2]:
            candidates.append((c, d, idx))
    candidates.sort(reverse=True)
    selected = []
    for c, d, idx in candidates:
        if not any(abs(d - sd) < min_separation for _, sd, _ in selected):
            selected.append((c, d, idx))
        if len(selected) >= top_k:
            break
    return selected


def get_tangent_at_arc(ordered_arc, deltas, target_arc):
    """Finds the dx,dy at the arc position closest to target_arc."""
    best_idx, best_diff = None, float('inf')
    for idx, d in ordered_arc:
        diff = abs(d - target_arc)
        if diff < best_diff:
            best_diff, best_idx = diff, idx
    return deltas[best_idx][0], deltas[best_idx][1]  # dx, dy


def check_tangent_reversal(ordered_arc, deltas, peak_arc, offset):
    dx_before, dy_before = get_tangent_at_arc(ordered_arc, deltas, peak_arc - offset)
    dx_after, dy_after = get_tangent_at_arc(ordered_arc, deltas, peak_arc + offset)

    dx_inversion = (dx_before > 0) != (dx_after > 0) and abs(dx_before) > 0.5 and abs(dx_after) > 0.5
    dy_inversion = (dy_before > 0) != (dy_after > 0) and abs(dy_before) > 0.5 and abs(dy_after) > 0.5

    return dx_inversion, dy_inversion


def main():
    overall_results = []

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
            deltas = compute_curvature_and_deltas(main_contour, ordered)
            peaks = find_top_local_peaks(ordered, deltas, top_k=3)

            confirmed = case["confirmed_cuff"][side_name]

            print(f"\n{side_name}")
            signature_hit_correct = False
            for rank, (c, d, idx) in enumerate(peaks, 1):
                pt = tuple(int(v) for v in main_contour[idx][0])
                dx_inv, dy_inv = check_tangent_reversal(ordered, deltas, d, TANGENT_CHECK_OFFSET_PX)
                signature = dx_inv or dy_inv
                is_confirmed = np.hypot(pt[0] - confirmed[0], pt[1] - confirmed[1]) < 15

                if signature and is_confirmed:
                    signature_hit_correct = True

                print(f"  Candidate #{rank}")
                print(f"    arc: {d:.1f}   pos: {pt}   curvature: {c:.1f}°")
                print(f"    dx inversion: {'YES' if dx_inv else 'NO'}   "
                      f"dy inversion: {'YES' if dy_inv else 'NO'}   "
                      f"signature: {'YES' if signature else 'NO'}")
                print(f"    confirmed cuff match: {'YES' if is_confirmed else 'NO'}")

            overall_results.append((case["label"], side_name, signature_hit_correct))

    print(f"\n\n{'='*75}")
    print("CANDIDATE SIGNATURE RESULT")
    print('='*75)
    correct = sum(1 for _, _, ok in overall_results if ok)
    for label, side, ok in overall_results:
        print(f"  {label:15s} {side:6s}  {'✅' if ok else '❌'}")
    print(f"\nTOTAL: {correct}/{len(overall_results)}")


if __name__ == "__main__":
    main()