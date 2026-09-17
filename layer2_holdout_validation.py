"""
Held-out validation of the frozen, combined Layer 2 mechanism —
bilateral collision + region exclusion + point exclusion — against
garments NEVER used to develop any part of it.

Zero changes to constants or logic. This measures generalization,
not development performance.

Run: python layer2_holdout_validation.py
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
POINT_EXCLUSION_RADIUS_PX = 15
COLLISION_RADIUS_PX = 15
REGION_EXCLUSION_RADIUS_PX = 100

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"

# Every item_id used anywhere in this entire investigation, excluded
DEVELOPMENT_ITEM_IDS = {
    "ac7fffb7-2e3a-40f8-894a-0e85fc245373",   # polo
    "25989827-db9d-4d65-90b3-12f5357b0b44",   # hanging_shirt
    "2ea4bb57-a358-4349-8aaf-204ded0772ab",   # rugby
    "6a53c00f-4f84-4e64-ad3b-5d2753504481",   # dress_shirt
    "bb7763f6-f1df-4b26-9522-94299149bb5e",
    "f2944c50-5757-4abc-81a6-c1380f8419a0",
    "d20577a1-01e9-4d6f-87a1-5ca89c3badba",
    "52f5fd29-d817-49bb-a0c8-ba181ba47157",
    "cb13467e-585f-4a4f-82fe-bac29082424e",
    "ded59bd3-57a1-43b1-888a-867e4362d2d8",
    "c93a3138-f40f-4e91-8d91-be3ff1c3065f",
    "5d19a29d-a7fa-4579-a1ba-2e374c3fd410",
    "fe419f35-3b62-43b9-842c-8883651e9c4f",
    "62930780-680d-4653-a8fa-daa7a6900617",
    "bd85602f-8c93-4d2c-99e9-7eafdb6da425",
    "2a32b5bb-c3ea-48f9-ac8d-5849eee866ba",
}

KNOWN_HEM_CORNERS = {}  # deliberately empty — held-out items have 
                          # no pre-known hem coordinates, testing 
                          # the mechanism's real behavior without 
                          # dataset-specific fixtures


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255


def find_underarm_indices(main_contour, w_img):
    hull_indices = cv2.convexHull(main_contour, returnPoints=False)
    try:
        defects = cv2.convexityDefects(main_contour, hull_indices)
    except cv2.error:
        return None, None
    if defects is None:
        return None, None
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
    if not left_candidates or not right_candidates:
        return None, None
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


def resolve_combined(left_candidates, right_candidates):
    li, ri = 0, 0
    excluded_regions_for_left = []
    excluded_regions_for_right = []

    while li < len(left_candidates) and ri < len(right_candidates):
        lc, ld, lpt = left_candidates[li]
        rc, rd, rpt = right_candidates[ri]

        if any(np.hypot(lpt[0]-r[0], lpt[1]-r[1]) < REGION_EXCLUSION_RADIUS_PX for r in excluded_regions_for_left):
            li += 1
            continue
        if any(np.hypot(rpt[0]-r[0], rpt[1]-r[1]) < REGION_EXCLUSION_RADIUS_PX for r in excluded_regions_for_right):
            ri += 1
            continue

        same_point_collision = np.hypot(lpt[0]-rpt[0], lpt[1]-rpt[1]) < COLLISION_RADIUS_PX

        if not same_point_collision:
            return (lpt, lc, ld), (rpt, rc, rd)

        if ld <= rd:
            excluded_regions_for_right.append(rpt)
            ri += 1
        else:
            excluded_regions_for_left.append(lpt)
            li += 1

    left_result = left_candidates[li] if li < len(left_candidates) else None
    right_result = right_candidates[ri] if ri < len(right_candidates) else None
    return (
        (left_result[2], left_result[0], left_result[1]) if left_result else None,
        (right_result[2], right_result[0], right_result[1]) if right_result else None
    )


def main():
    result = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/?user_id={AVATAR_USER_ID}"
    ).json()["data"]

    all_items = result.get("items", [])
    holdout_items = [
        item for item in all_items
        if item["item_id"] not in DEVELOPMENT_ITEM_IDS
        and item.get("category") == "top"
        and not item.get("is_archived", False)
    ]

    print(f"Found {len(holdout_items)} held-out items (never used to develop Layer 2)\n")

    for item in holdout_items:
        print(f"\n{'='*75}")
        print(f"{item['item_id']} — {item.get('item_type', 'unknown')}")
        print('='*75)

        image_bytes = requests.get(item["clean_image_url"]).content
        binary_mask = get_binary_mask(image_bytes)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_contour = max(contours, key=cv2.contourArea)
        h_img, w_img = binary_mask.shape

        left_idx, right_idx = find_underarm_indices(main_contour, w_img)
        if left_idx is None:
            print("  Could not find both underarm points — skipping")
            continue

        underarms = [tuple(int(v) for v in main_contour[left_idx][0]),
                     tuple(int(v) for v in main_contour[right_idx][0])]

        left_c = get_filtered_candidates(main_contour, left_idx, h_img, underarms)
        right_c = get_filtered_candidates(main_contour, right_idx, h_img, underarms)

        left_result, right_result = resolve_combined(left_c, right_c)

        for side_name, result_pt in [("LEFT", left_result), ("RIGHT", right_result)]:
            if result_pt is None:
                print(f"  {side_name}: NO VALID CANDIDATE")
                continue
            pt, c, d = result_pt
            print(f"  {side_name}: chosen={pt}  curvature={c:.1f}°  arc={d:.1f}px")

        print(f"\n  >>> RECORD: does LEFT/RIGHT land at plausible cuff locations?")


if __name__ == "__main__":
    main()