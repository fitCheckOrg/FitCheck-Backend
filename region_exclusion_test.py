"""
Tests REGION-based exclusion: instead of a growing list of exact
coordinates, exclude any candidate within REGION_RADIUS_PX of an
already-known landmark (hem corner, or LEFT's own resolved cuff
once found). Tests whether this clears hanging-shirt LEFT's whole
multi-point cluster at once, without needing a third/fourth exact
coordinate.

Only the hem corner (366,458) is used as a pre-known fixture entry
here — LEFT's cuff itself is NOT hardcoded, since in production we
won't know it in advance; instead, once LEFT resolves to its own
cuff, that result is used to exclude RIGHT's search within the
same region.

Run: python region_exclusion_test.py
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
MIN_PEAK_SEPARATION = 10
TOP_K = 8
COLLAR_ZONE_Y_PCT = 15
POINT_EXCLUSION_RADIUS_PX = 15   # for known fixture points (hem corners)
REGION_EXCLUSION_RADIUS_PX = 100  # NEW — excludes whole neighborhood 
                                     # around the OTHER side's resolved cuff

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"

KNOWN_HEM_CORNERS = {
    "polo": [(378, 445), (93, 442)],
    "hanging_shirt": [(366, 458)],  # hem corner only — NOT the cuff cluster points
}

TEST_CASES = [
    {"item_id": "ac7fffb7-2e3a-40f8-894a-0e85fc245373", "label": "polo",
     "confirmed_cuff": {"LEFT": (31, 154), "RIGHT": (439, 142)}},
    {"item_id": "25989827-db9d-4d65-90b3-12f5357b0b44", "label": "hanging_shirt",
     "confirmed_cuff": {"LEFT": (26, 457), "RIGHT": (442, 447)}},
    {"item_id": "2ea4bb57-a358-4349-8aaf-204ded0772ab", "label": "rugby",
     "confirmed_cuff": {"LEFT": (51, 488), "RIGHT": (430, 459)}},
    {"item_id": "6a53c00f-4f84-4e64-ad3b-5d2753504481", "label": "dress_shirt",
     "confirmed_cuff": {"LEFT": (54, 633), "RIGHT": (166, 660)},
     "out_of_scope_sides": ["RIGHT"]},
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


def get_filtered_candidates(main_contour, underarm_idx, h_img, point_landmarks):
    forward, backward = contour_arc_distances(main_contour, underarm_idx)
    net_y_fwd = get_net_y_change(main_contour, forward, NET_Y_WINDOW_PX)
    net_y_bwd = get_net_y_change(main_contour, backward, NET_Y_WINDOW_PX)
    sleeve_arc = forward if abs(net_y_fwd) < abs(net_y_bwd) else backward
    ordered = sorted(sleeve_arc.items(), key=lambda kv: kv[1])
    curvatures = compute_curvature(main_contour, ordered)
    peaks = find_top_k_peaks(ordered, curvatures, TOP_K)
    filtered = []
    for c, d, idx in peaks:
        pt = tuple(int(v) for v in main_contour[idx][0])
        y_pct = pt[1] / h_img * 100
        is_known = any(np.hypot(pt[0]-lm[0], pt[1]-lm[1]) < POINT_EXCLUSION_RADIUS_PX
                        for lm in point_landmarks)
        is_collar = y_pct < COLLAR_ZONE_Y_PCT
        if not is_known and not is_collar:
            filtered.append((c, d, pt))
    return filtered


def main():
    correct = 0
    total = 0

    for case in TEST_CASES:
        out_of_scope = case.get("out_of_scope_sides", [])
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
        h_img, w_img = binary_mask.shape

        left_idx, right_idx = find_underarm_indices(main_contour, w_img)
        underarms = [tuple(int(v) for v in main_contour[left_idx][0]),
                     tuple(int(v) for v in main_contour[right_idx][0])]
        point_landmarks = underarms + KNOWN_HEM_CORNERS.get(case["label"], [])

        if "RIGHT" in out_of_scope:
            left_c = get_filtered_candidates(main_contour, left_idx, h_img, point_landmarks)
            c, d, pt = left_c[0]
            confirmed = case["confirmed_cuff"]["LEFT"]
            dist = np.hypot(pt[0]-confirmed[0], pt[1]-confirmed[1])
            match = dist < 15
            total += 1
            if match: correct += 1
            print(f"  LEFT only: chosen={pt}  confirmed={confirmed}  dist={dist:.1f}px  {'✅' if match else '❌'}")
            continue

        # STEP 1: resolve LEFT first, independently, using point-level exclusion only
        left_c = get_filtered_candidates(main_contour, left_idx, h_img, point_landmarks)
        left_chosen = left_c[0]  # (curvature, arc, point)
        left_pt = left_chosen[2]

        # STEP 2: resolve RIGHT, now ALSO excluding the REGION around LEFT's resolved cuff
        right_c_raw = get_filtered_candidates(main_contour, right_idx, h_img, point_landmarks)
        right_c = [(c, d, pt) for c, d, pt in right_c_raw
                   if np.hypot(pt[0]-left_pt[0], pt[1]-left_pt[1]) >= REGION_EXCLUSION_RADIUS_PX]
        right_chosen = right_c[0] if right_c else None

        confirmed_left = case["confirmed_cuff"]["LEFT"]
        confirmed_right = case["confirmed_cuff"]["RIGHT"]

        for side_name, chosen, confirmed in [
            ("LEFT", left_chosen, confirmed_left),
            ("RIGHT", right_chosen, confirmed_right)
        ]:
            total += 1
            if chosen is None:
                print(f"  {side_name}: NO VALID CANDIDATE ❌")
                continue
            c, d, pt = chosen
            dist = np.hypot(pt[0]-confirmed[0], pt[1]-confirmed[1])
            match = dist < 15
            if match: correct += 1
            print(f"  {side_name}: chosen={pt} (curv={c:.1f}°, arc={d:.1f}px)  "
                  f"confirmed={confirmed}  dist={dist:.1f}px  {'✅' if match else '❌'}")

    print(f"\n\n{'='*75}")
    print(f"FINAL: {correct}/{total}")
    print('='*75)


if __name__ == "__main__":
    main()