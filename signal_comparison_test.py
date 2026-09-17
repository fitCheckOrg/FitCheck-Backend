"""
Tests BOTH candidate signals — x-extremum (original approach) and
y-extremum (new hypothesis from dress-shirt finding) — against all
four garments' independently confirmed cuff coordinates.

Hypothesis: sleeve orientation determines which signal applies.
Extended/short sleeves (polo) → x-extremum. Hanging/long sleeves
(hanging shirt, rugby, dress shirt) → y-extremum. Neither signal
alone may be universal.

Run: python signal_comparison_test.py
"""

import io
import requests
import numpy as np
import cv2
from PIL import Image

ALPHA_THRESHOLD = 10
MIN_DEFECT_DEPTH = 1000
NET_Y_WINDOW_PX = 300
SEARCH_CAP = 350  # generous, covers all confirmed cuff arc-depths tonight

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"

# Confirmed ground truth from tonight's full investigation
CONFIRMED_CUFFS = {
    "ac7fffb7-2e3a-40f8-894a-0e85fc245373": {"label": "polo", "LEFT": (31, 154), "RIGHT": (439, 142)},
    "25989827-db9d-4d65-90b3-12f5357b0b44": {"label": "hanging_shirt", "LEFT": (26, 457), "RIGHT": (442, 447)},
    "2ea4bb57-a358-4349-8aaf-204ded0772ab": {"label": "rugby", "LEFT": (51, 488), "RIGHT": (430, 459)},
    "6a53c00f-4f84-4e64-ad3b-5d2753504481": {"label": "dress_shirt", "LEFT": (54, 633), "RIGHT": (166, 660)},
}


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


def find_y_extremum(contour, arc_dict, underarm_x, search_cap):
    """Candidate signal A: point of maximum y (deepest reach)."""
    best_idx, best_d, best_y = None, None, -1
    for idx, d in arc_dict.items():
        if d > search_cap:
            continue
        y = int(contour[idx][0][1])
        if y > best_y:
            best_y, best_idx, best_d = y, idx, d
    return tuple(int(v) for v in contour[best_idx][0]) if best_idx else None


def find_x_extremum(contour, arc_dict, underarm_x, search_cap):
    """Candidate signal B: point of maximum |x - underarm_x| (farthest lateral reach)."""
    best_idx, best_d, best_disp = None, None, -1
    for idx, d in arc_dict.items():
        if d > search_cap:
            continue
        x = int(contour[idx][0][0])
        disp = abs(x - underarm_x)
        if disp > best_disp:
            best_disp, best_idx, best_d = disp, idx, d
    return tuple(int(v) for v in contour[best_idx][0]) if best_idx else None


def distance(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])


def main():
    results = []

    for item_id, info in CONFIRMED_CUFFS.items():
        print(f"\n{'='*75}")
        print(f"{item_id} — {info['label']}")
        print('='*75)

        result_data = requests.get(
            f"http://127.0.0.1:8000/api/wardrobe/{item_id}?user_id={AVATAR_USER_ID}"
        ).json()["data"]
        image_bytes = requests.get(result_data["clean_image_url"]).content

        binary_mask = get_binary_mask(image_bytes)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_contour = max(contours, key=cv2.contourArea)
        w_img = binary_mask.shape[1]

        left_idx, right_idx = find_underarm_indices(main_contour, w_img)

        for side_name, underarm_idx in [("LEFT", left_idx), ("RIGHT", right_idx)]:
            underarm_x = int(main_contour[underarm_idx][0][0])
            forward, backward = contour_arc_distances(main_contour, underarm_idx)

            net_y_fwd = get_net_y_change(main_contour, forward, NET_Y_WINDOW_PX)
            net_y_bwd = get_net_y_change(main_contour, backward, NET_Y_WINDOW_PX)
            sleeve_arc = forward if abs(net_y_fwd) < abs(net_y_bwd) else backward

            y_extremum = find_y_extremum(main_contour, sleeve_arc, underarm_x, SEARCH_CAP)
            x_extremum = find_x_extremum(main_contour, sleeve_arc, underarm_x, SEARCH_CAP)

            confirmed = info[side_name]
            y_dist = distance(y_extremum, confirmed) if y_extremum else None
            x_dist = distance(x_extremum, confirmed) if x_extremum else None

            print(f"\n  {side_name} — confirmed cuff: {confirmed}")
            print(f"    Y-extremum candidate: {y_extremum}  "
                  f"distance_from_truth={y_dist:.1f}px" if y_dist is not None else
                  f"    Y-extremum candidate: {y_extremum}  distance_from_truth=N/A")
            print(f"    X-extremum candidate: {x_extremum}  "
                  f"distance_from_truth={x_dist:.1f}px" if x_dist is not None else
                  f"    X-extremum candidate: {x_extremum}  distance_from_truth=N/A")

            results.append({
                "garment": info["label"], "side": side_name,
                "y_dist": y_dist, "x_dist": x_dist
            })

    print(f"\n\n{'='*75}")
    print("SUMMARY — which signal is closer to ground truth, per case")
    print('='*75)
    for r in results:
        if r["y_dist"] is None or r["x_dist"] is None:
            print(f"  {r['garment']:15s} {r['side']:6s}  Y_dist={r['y_dist']}  "
                  f"X_dist={r['x_dist']}  → INCOMPLETE, needs investigation")
            continue
        winner = "Y-extremum" if r["y_dist"] < r["x_dist"] else "X-extremum"
        print(f"  {r['garment']:15s} {r['side']:6s}  Y_dist={r['y_dist']:6.1f}  "
              f"X_dist={r['x_dist']:6.1f}  → {winner}")

if __name__ == "__main__":
    main()