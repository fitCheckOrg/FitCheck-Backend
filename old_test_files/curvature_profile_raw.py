"""
Raw curvature profile diagnostic — zero threshold tuning, zero
cuff-selection logic. Exposes the actual curvature signal along
each sleeve direction so we can see whether v5's stopping point
was a real geometric event or a spurious spike.

Reuses v4's confirmed direction classification (which arc is
sleeve for each side) and v5's curvature computation, unchanged.
Only adds: full visibility into the raw numbers.

Run: python curvature_profile_raw.py
"""

import io
import requests
import numpy as np
import cv2
from PIL import Image

ALPHA_THRESHOLD = 10
MIN_DEFECT_DEPTH = 1000
CURVATURE_WINDOW = 8
CLASSIFICATION_WINDOW_PX = 60
MIN_ARC_BEFORE_PEAK_SEARCH = 30
PRINT_INTERVAL_PX = 10   # print roughly every 10px of arc, not every point

TEST_CASES = [
    {"item_id": "25989827-db9d-4d65-90b3-12f5357b0b44", "label": "hanging_sleeve_shirt"},
    {"item_id": "ac7fffb7-2e3a-40f8-894a-0e85fc245373", "label": "polo_baseline"},
]

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


def classify_direction(contour, underarm_x, arc_dict, window_px):
    max_disp = 0
    for idx, d in arc_dict.items():
        if d > window_px:
            continue
        x = contour[idx][0][0]
        max_disp = max(max_disp, abs(int(x) - underarm_x))
    return max_disp


def compute_curvature_at_indices(contour, arc_dict, window=CURVATURE_WINDOW):
    ordered = sorted(arc_dict.items(), key=lambda kv: kv[1])
    indices = [idx for idx, d in ordered]
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
        if norm1 == 0 or norm2 == 0:
            curvatures[indices[i]] = 0.0
            continue
        cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        curvatures[indices[i]] = np.degrees(np.arccos(cos_angle))
    return curvatures


def print_curvature_profile(contour, sleeve_arc, label):
    curvatures = compute_curvature_at_indices(contour, sleeve_arc)
    ordered = sorted(sleeve_arc.items(), key=lambda kv: kv[1])

    print(f"\n{'='*70}")
    print(f"CURVATURE PROFILE — {label}")
    print(f"{'='*70}")

    last_printed = -PRINT_INTERVAL_PX
    first_spike_reported = False

    for idx, d in ordered:
        if d < MIN_ARC_BEFORE_PEAK_SEARCH:
            continue
        if d - last_printed < PRINT_INTERVAL_PX:
            continue
        last_printed = d

        x, y = contour[idx][0]
        curvature = curvatures.get(idx, 0.0)
        flag = "  <<< >50°" if curvature > 50.0 else ""
        if curvature > 50.0 and not first_spike_reported:
            flag += "  [FIRST SPIKE]"
            first_spike_reported = True

        print(f"arc={d:6.1f}px  idx={idx:4d}  pos=({int(x):3d},{int(y):3d})  "
              f"curvature={curvature:6.1f}°{flag}")


def main():
    for case in TEST_CASES:
        print(f"\n\n{'#'*70}")
        print(f"# {case['item_id']} — {case['label']}")
        print(f"{'#'*70}")

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
            underarm_x = int(main_contour[underarm_idx][0][0])
            forward, backward = contour_arc_distances(main_contour, underarm_idx)

            fwd_disp = classify_direction(main_contour, underarm_x, forward, CLASSIFICATION_WINDOW_PX)
            bwd_disp = classify_direction(main_contour, underarm_x, backward, CLASSIFICATION_WINDOW_PX)

            sleeve_arc = forward if fwd_disp > bwd_disp else backward
            sleeve_direction = "FORWARD" if fwd_disp > bwd_disp else "BACKWARD"

            print_curvature_profile(main_contour, sleeve_arc,
                                      f"{case['label']} — {side_name} ({sleeve_direction})")


if __name__ == "__main__":
    main()