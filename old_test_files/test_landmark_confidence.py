"""
Landmark confidence scoring — live test against the 4 real garments.

Runs the complete pipeline fresh (underarm detection, path split,
curvature, peak-finding) then applies landmark_confidence() to
every peak, checking specifically:
  - Does A#1 (known real corner, no pair) score well despite having
    no symmetric counterpart?
  - Does the black shirt's 155.4° mid-sleeve false positive score
    LOWER than genuine hem/cuff corners, despite having the highest
    raw curvature in the whole dataset?

Run: python test_landmark_confidence.py
"""

import sys
import io
import requests
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, ".")
from core.storage.supabase_client import supabase
from workers.wardrobe.landmark_confidence import landmark_confidence

ALPHA_THRESHOLD = 10
MIN_DEFECT_DEPTH = 1000
CURVATURE_WINDOW = 8
MIN_PEAK_SEPARATION = 15
TOP_K_PEAKS = 20  # keep at 20, per tonight's finding that 8 was 
                    # sometimes too aggressive

TEST_CASES = [
    {"item_id": "25989827-db9d-4d65-90b3-12f5357b0b44", "label": "hanger_shirt"},
    {"item_id": "ba2e5aa1-bfb5-462f-aaad-fe378f658c42", "label": "long_sleeve_hem_failure"},
    {"item_id": "ac7fffb7-2e3a-40f8-894a-0e85fc245373", "label": "polo_baseline"},
    {"item_id": "bb7763f6-f1df-4b26-9522-94299149bb5e", "label": "new_untested_polo"},
]


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255


def find_underarms(main_contour, w_img):
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


def build_event_records(path, peaks, path_id, w_img, h_img, left_pt, right_pt):
    records = []
    for rank, (curvature, idx) in enumerate(peaks, 1):
        pt = path[idx][0]
        x, y = float(pt[0]), float(pt[1])
        dist_left = np.hypot(x - left_pt[0], y - left_pt[1]) / w_img * 100
        dist_right = np.hypot(x - right_pt[0], y - right_pt[1]) / w_img * 100
        records.append({
            "path_id": path_id, "rank": rank, "peak_idx": idx,
            "curvature": round(float(curvature), 1),
            "x_pct": round(x / w_img * 100, 1), "y_pct": round(y / h_img * 100, 1),
            "dist_to_left_underarm_pct": round(dist_left, 1),
            "dist_to_right_underarm_pct": round(dist_right, 1),
        })
    return records


def main():
    for case in TEST_CASES:
        print(f"\n{'='*80}")
        print(f"{case['item_id']} — {case['label']}")
        print('='*80)

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

        all_events = []
        curvature_profiles = {}
        for path_name, path in [("A", path_a), ("B", path_b)]:
            curv = compute_curvature_profile(path)
            curvature_profiles[path_name] = curv
            peaks = find_separated_peaks(curv, MIN_PEAK_SEPARATION, TOP_K_PEAKS)
            all_events.extend(build_event_records(path, peaks, path_name, w_img, h_img, left_pt, right_pt))

        scored = []
        for event in all_events:
            profile = curvature_profiles[event["path_id"]]
            score, reasons = landmark_confidence(event, all_events, profile)
            scored.append((score, event, reasons))

        scored.sort(reverse=True, key=lambda x: x[0])

        print(f"\n  Ranked by CONFIDENCE SCORE (not raw curvature):")
        for score, event, reasons in scored[:10]:
            print(f"\n    score={score}  {event['path_id']}#{event['rank']}  "
                  f"curvature={event['curvature']}°  pos=({event['x_pct']}%, {event['y_pct']}%)")
            for r in reasons:
                print(f"      - {r}")


if __name__ == "__main__":
    main()