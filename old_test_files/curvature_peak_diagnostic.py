"""
Curvature peak-location diagnostic — the actual next experiment,
not another aggregate statistic.

Finds LOCAL MAXIMA in the curvature profile (not just top-K raw
values, which would let one sharp corner dominate the list with
near-duplicate adjacent points), enforces minimum separation along
the contour so each reported peak is a genuinely distinct event,
and prints exact contour-index + position + curvature for each —
enough to determine whether peaks land on cuffs/hem-corners, and
enough to mechanically explain the stray interior-dot anomaly
rather than speculate about it.

Run: python curvature_peak_diagnostic.py
"""

import sys
import io
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw

sys.path.insert(0, ".")
from core.storage.supabase_client import supabase

ALPHA_THRESHOLD = 10
MIN_DEFECT_DEPTH = 1000
CURVATURE_WINDOW = 8
MIN_PEAK_SEPARATION = 15  # minimum contour-index distance between 
                            # two reported peaks — prevents adjacent 
                            # points on the same corner from counting 
                            # as separate events
TOP_K_PEAKS = 8

TEST_CASES = [
    {"item_id": "25989827-db9d-4d65-90b3-12f5357b0b44", "label": "hanger_shirt"},
    {"item_id": "ba2e5aa1-bfb5-462f-aaad-fe378f658c42", "label": "long_sleeve_hem_failure"},
    {"item_id": "ac7fffb7-2e3a-40f8-894a-0e85fc245373", "label": "polo_baseline"},
]


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255, image


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
    left_idx = left_candidates[0][1] if left_candidates else None
    right_idx = right_candidates[0][1] if right_candidates else None
    return left_idx, right_idx


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
    """
    Local-maxima detection with minimum index separation. A point
    counts as a candidate peak only if it's higher than both
    immediate neighbors. Candidates are then greedily selected
    highest-first, skipping any candidate too close (along the
    contour) to an already-selected peak — this is what prevents
    one physical corner's several adjacent high-curvature samples
    from being reported as multiple distinct "events."
    """
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


def underarm_distance_pct(idx, left_idx, right_idx, path_len_total):
    """Distance along the contour to the nearer underarm point, 
    as a percentage of total contour length — cheap proxy, not 
    real geometric distance, but enough to see if peaks cluster 
    near or far from known underarm locations."""
    dist_left = abs(idx - left_idx) if left_idx is not None else float('inf')
    dist_right = abs(idx - right_idx) if right_idx is not None else float('inf')
    nearer = min(dist_left, dist_right)
    return round(nearer / path_len_total * 100, 1)


def visualize_peaks(image, path, peaks, w_img, label_prefix):
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    pts = [(int(p[0][0]), int(p[0][1])) for p in path]
    draw.line(pts, fill=(80, 80, 80), width=1)

    for rank, (curvature, idx) in enumerate(peaks, 1):
        x, y = int(path[idx][0][0]), int(path[idx][0][1])
        draw.ellipse([x-9, y-9, x+9, y+9], outline=(255, 255, 0), width=2)
        draw.text((x+11, y-6), f"{label_prefix}{rank}", fill=(255, 255, 0))

    return canvas


def main():
    for case in TEST_CASES:
        print(f"\n{'='*70}")
        print(f"{case['item_id']} — {case['label']}")
        print('='*70)

        result = supabase.table("closet_items")\
            .select("clean_image_url").eq("item_id", case["item_id"]).single().execute()
        image_bytes = requests.get(result.data["clean_image_url"]).content

        binary_mask, original = get_binary_mask(image_bytes)
        h_img, w_img = binary_mask.shape

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_contour = max(contours, key=cv2.contourArea)
        print(f"  Total contours found (RETR_EXTERNAL): {len(contours)}")
        print(f"  Main contour point count: {len(main_contour)}")

        left_idx, right_idx = find_underarm_indices(main_contour, w_img)
        if left_idx is None or right_idx is None:
            print("  Could not find both underarm points — skipping")
            continue

        path_a, path_b = split_contour_paths(main_contour, left_idx, right_idx)

        canvas = original.convert("RGB").copy()

        for path_name, path in [("A", path_a), ("B", path_b)]:
            curv = compute_curvature_profile(path)
            peaks = find_separated_peaks(curv, MIN_PEAK_SEPARATION, TOP_K_PEAKS)

            print(f"\n  PATH_{path_name} — {len(peaks)} separated peaks found:")
            for rank, (curvature, idx) in enumerate(peaks, 1):
                pt = path[idx][0]
                x_pct = round(pt[0] / w_img * 100, 1)
                y_pct = round(pt[1] / h_img * 100, 1)
                dist_pct = underarm_distance_pct(idx, 0, len(path)-1, len(path))
                print(f"    #{rank}: curvature={curvature:.1f}°  "
                      f"contour_idx={idx}  position=({x_pct}%, {y_pct}%)  "
                      f"path_end_distance={dist_pct}%")

            canvas = visualize_peaks(canvas, path, peaks, w_img, path_name)

        filename = f"peaks_{case['label']}.png"
        canvas.save(filename)
        print(f"\n  Saved: {filename}")


if __name__ == "__main__":
    main()