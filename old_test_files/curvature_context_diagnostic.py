"""
Curvature peak context + pairing diagnostic.

Extends the peak-location diagnostic with the requested context
fields per peak, plus a genuine test of the pairing hypothesis:
do legitimate landmark pairs (left cuff / right cuff, left hem
corner / right hem corner) share similar Y and mirror X-position
around the garment centerline, in a way that distinguishes them
from non-paired events like the black shirt's mid-sleeve bend?

Does NOT implement a classifier. Reports context + candidate pairs
for manual inspection against the three known cases.

Run: python curvature_context_diagnostic.py
"""

import sys
import io
import requests
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, ".")
from core.storage.supabase_client import supabase

ALPHA_THRESHOLD = 10
MIN_DEFECT_DEPTH = 1000
CURVATURE_WINDOW = 8
MIN_PEAK_SEPARATION = 15
TOP_K_PEAKS = 20
PAIR_Y_TOLERANCE_PCT = 8      # peaks within this Y% of each other 
                                # are candidates for pairing
PAIR_X_SYMMETRY_TOLERANCE_PCT = 10  # how close to true mirror 
                                       # (100 - x) counts as symmetric

# TEST_CASES = [
#     {"item_id": "25989827-db9d-4d65-90b3-12f5357b0b44", "label": "hanger_shirt"},
#     {"item_id": "ba2e5aa1-bfb5-462f-aaad-fe378f658c42", "label": "long_sleeve_hem_failure"},
#     {"item_id": "ac7fffb7-2e3a-40f8-894a-0e85fc245373", "label": "polo_baseline"},
# ]
TEST_CASES = [
    {"item_id": "bb7763f6-f1df-4b26-9522-94299149bb5e", "label": "NEW_untested_polo"},
]

def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255


def find_underarms(main_contour, w_img):
    """Returns (left_idx, left_point, right_idx, right_point) — 
    now returning actual pixel positions too, needed for real 
    distance calculations, not just indices."""
    hull_indices = cv2.convexHull(main_contour, returnPoints=False)
    try:
        defects = cv2.convexityDefects(main_contour, hull_indices)
    except cv2.error:
        return None, None, None, None
    if defects is None:
        return None, None, None, None

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
        return None, None, None, None

    left_idx = left_candidates[0][1]
    right_idx = right_candidates[0][1]
    left_point = tuple(int(v) for v in main_contour[left_idx][0])
    right_point = tuple(int(v) for v in main_contour[right_idx][0])
    return left_idx, left_point, right_idx, right_point


def split_contour_paths(main_contour, left_idx, right_idx):
    lo, hi = sorted([left_idx, right_idx])
    path_a = main_contour[lo:hi+1]
    path_b = np.concatenate([main_contour[hi:], main_contour[:lo+1]])
    return path_a, path_b


def compute_curvature_profile(path):
    pts = path[:, 0, :].astype(float)
    n = len(pts)
    curvatures = []
    for i in range(n):
        prev_i = max(0, i - CURVATURE_WINDOW)
        next_i = min(n - 1, i + CURVATURE_WINDOW)
        if next_i - prev_i < 2:
            curvatures.append(0.0)
            continue
        v1 = pts[i] - pts[prev_i]
        v2 = pts[next_i] - pts[i]
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            curvatures.append(0.0)
            continue
        cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        curvatures.append(np.degrees(np.arccos(cos_angle)))
    return np.array(curvatures)


def find_separated_peaks(curvatures, min_separation, top_k):
    n = len(curvatures)
    candidates = []
    for i in range(1, n - 1):
        if curvatures[i] > curvatures[i-1] and curvatures[i] > curvatures[i+1]:
            candidates.append((curvatures[i], i))
    candidates.sort(reverse=True)
    selected = []
    for curvature, idx in candidates:
        too_close = any(abs(idx - sel_idx) < min_separation for _, sel_idx in selected)
        if not too_close:
            selected.append((curvature, idx))
        if len(selected) >= top_k:
            break
    return selected


def build_peak_records(path, peaks, path_id, w_img, h_img, left_pt, right_pt):
    """Full context per peak, per the requested field list."""
    records = []
    underarm_span_x = right_pt[0] - left_pt[0]
    underarm_avg_y = (left_pt[1] + right_pt[1]) / 2

    for rank, (curvature, idx) in enumerate(peaks, 1):
        pt = path[idx][0]
        x, y = float(pt[0]), float(pt[1])

        dist_left = np.hypot(x - left_pt[0], y - left_pt[1])
        dist_right = np.hypot(x - right_pt[0], y - right_pt[1])

        x_rel_span = ((x - left_pt[0]) / underarm_span_x * 100) if underarm_span_x != 0 else None
        y_rel_underarm = y - underarm_avg_y

        records.append({
            "path_id": path_id,
            "rank": rank,
            "curvature": round(float(curvature), 1),
            "x_pct": round(x / w_img * 100, 1),
            "y_pct": round(y / h_img * 100, 1),
            "dist_to_left_underarm_pct": round(dist_left / w_img * 100, 1),
            "dist_to_right_underarm_pct": round(dist_right / w_img * 100, 1),
            "x_relative_to_underarm_span_pct": round(x_rel_span, 1) if x_rel_span is not None else None,
            "y_relative_to_underarm_y_px_pct": round(y_rel_underarm / h_img * 100, 1),
            "arc_position_pct": round(idx / len(path) * 100, 1),
        })
    return records


def find_candidate_pairs(all_records):
    """
    Tests the pairing hypothesis directly: for every pair of peaks,
    check if they share similar Y and roughly mirror X around the
    50% centerline. Reports candidates for manual judgment — does
    NOT auto-classify anything as a confirmed pair.
    """
    pairs = []
    n = len(all_records)
    for i in range(n):
        for j in range(i + 1, n):
            r1, r2 = all_records[i], all_records[j]
            y_diff = abs(r1["y_pct"] - r2["y_pct"])
            if y_diff > PAIR_Y_TOLERANCE_PCT:
                continue

            mirror_x_of_r1 = 100 - r1["x_pct"]
            x_symmetry_diff = abs(mirror_x_of_r1 - r2["x_pct"])
            if x_symmetry_diff > PAIR_X_SYMMETRY_TOLERANCE_PCT:
                continue

            pairs.append({
                "peak_1": f"{r1['path_id']}#{r1['rank']} ({r1['x_pct']}%, {r1['y_pct']}%, curv={r1['curvature']})",
                "peak_2": f"{r2['path_id']}#{r2['rank']} ({r2['x_pct']}%, {r2['y_pct']}%, curv={r2['curvature']})",
                "y_diff_pct": round(y_diff, 1),
                "x_symmetry_diff_pct": round(x_symmetry_diff, 1),
            })
    return pairs


def main():
    for case in TEST_CASES:
        print(f"\n{'='*75}")
        print(f"{case['item_id']} — {case['label']}")
        print('='*75)

        result = supabase.table("closet_items")\
            .select("clean_image_url").eq("item_id", case["item_id"]).single().execute()
        image_bytes = requests.get(result.data["clean_image_url"]).content

        binary_mask = get_binary_mask(image_bytes)
        h_img, w_img = binary_mask.shape

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_contour = max(contours, key=cv2.contourArea)

        left_idx, left_pt, right_idx, right_pt = find_underarms(main_contour, w_img)
        if left_idx is None:
            print("  Could not find both underarm points — skipping")
            continue

        path_a, path_b = split_contour_paths(main_contour, left_idx, right_idx)

        all_records = []
        for path_name, path in [("A", path_a), ("B", path_b)]:
            curv = compute_curvature_profile(path)
            peaks = find_separated_peaks(curv, MIN_PEAK_SEPARATION, TOP_K_PEAKS)
            records = build_peak_records(path, peaks, path_name, w_img, h_img, left_pt, right_pt)
            all_records.extend(records)

            print(f"\n  PATH_{path_name}:")
            for r in records:
                print(f"    #{r['rank']}: curv={r['curvature']}°  pos=({r['x_pct']}%, {r['y_pct']}%)  "
                      f"dist_L={r['dist_to_left_underarm_pct']}%  dist_R={r['dist_to_right_underarm_pct']}%  "
                      f"x_rel_span={r['x_relative_to_underarm_span_pct']}%  "
                      f"y_rel_underarm={r['y_relative_to_underarm_y_px_pct']}%  "
                      f"arc_pos={r['arc_position_pct']}%")

        print(f"\n  CANDIDATE PAIRS (similar Y, mirrored X):")
        pairs = find_candidate_pairs(all_records)
        if not pairs:
            print("    None found within tolerance")
        for p in pairs:
            print(f"    {p['peak_1']}  <-->  {p['peak_2']}  "
                  f"(y_diff={p['y_diff_pct']}%, x_symmetry_diff={p['x_symmetry_diff_pct']}%)")


if __name__ == "__main__":
    main()