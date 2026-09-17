"""
Tests whether the internal ridge visible in the shoulder region
exists on BOTH sides with corresponding geometry — the key
discriminator between a real construction seam (bilateral,
expected) and incidental crease/lighting structure (likely
one-sided or inconsistent).

Steps, per the explicit plan:
1. Compute internal gradients, masked to garment interior only
   (outer silhouette explicitly suppressed via erosion).
2. Restrict search to a shoulder-region corridor on each side.
3. Trace connected ridge candidates via simple thinning/skeleton
   of the strong-gradient mask.
4. Measure length, orientation, and distance from underarm for
   each side's strongest candidate.
5. Report left vs right for direct comparison. No landmark
   promotion, no threshold tuning beyond what's needed to run
   the comparison once.

Run: python -m tryon.internal_ridge_bilateral_test
"""

import io
import os
import requests
import numpy as np
import cv2
from PIL import Image

from tryon.correspondence_shoulders_test import find_garment_underarms
from tryon.garment_shoulder_region_extraction import get_binary_mask

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
PLAIN_SHIRT_ID = "7e16734e-6cb2-4254-adf5-4695549ad5db"
OUTPUT_DIR = "outputs"
EROSION_PX = 8  # shrink garment mask inward to fully exclude outer silhouette
CORRIDOR_HALF_WIDTH = 90  # shoulder-region search corridor, per side


def get_interior_gradient(gray, mask):
    """Gradient magnitude, computed only where the mask (after
    erosion, to exclude the outer boundary entirely) is True."""
    kernel = np.ones((EROSION_PX, EROSION_PX), np.uint8)
    eroded_mask = cv2.erode(mask.astype(np.uint8), kernel) > 0

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mag[~eroded_mask] = 0
    return grad_mag, eroded_mask


def find_ridge_candidate(grad_mag, region_mask, y_range, x_range):
    """Within the given corridor, thresholds the strongest gradient
    pixels, finds connected components, and returns the largest
    one's bounding info (length, orientation) — the top ridge
    candidate for that side."""
    y0, y1 = y_range
    x0, x1 = x_range
    sub_grad = grad_mag[y0:y1, x0:x1]
    sub_mask = region_mask[y0:y1, x0:x1]

    vals = sub_grad[sub_mask]
    if len(vals) == 0:
        return None
    threshold = np.percentile(vals, 92)
    strong = (sub_grad > threshold) & sub_mask

    strong_u8 = (strong * 255).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(strong_u8, connectivity=8)

    if num_labels <= 1:
        return None

    # Largest component by area (excluding background label 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = np.argmax(areas) + 1
    component_mask = (labels == largest_idx)

    ys, xs = np.where(component_mask)
    if len(ys) < 5:
        return None

    length = np.hypot(ys.max() - ys.min(), xs.max() - xs.min())
    # Orientation via PCA on the component's points
    pts = np.column_stack([xs, ys]).astype(np.float64)
    pts -= pts.mean(axis=0)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    angle_deg = np.degrees(np.arctan2(principal[1], principal[0])) % 180

    return {
        "area": int(areas[largest_idx - 1]),
        "length": float(length),
        "angle_deg": float(angle_deg),
        "centroid_global": (float(centroids[largest_idx][0] + x0), float(centroids[largest_idx][1] + y0)),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    garment_resp = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{PLAIN_SHIRT_ID}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    garment_bytes = requests.get(garment_resp["clean_image_url"]).content
    garment_mask, garment_img = get_binary_mask(garment_bytes)
    gw, gh = garment_img.size
    gray = cv2.cvtColor(np.array(garment_img.convert("RGB")), cv2.COLOR_RGB2GRAY)

    left_ua, right_ua = find_garment_underarms(garment_mask, gw)
    print(f"Underarms: L={left_ua}  R={right_ua}")

    grad_mag, eroded_mask = get_interior_gradient(gray, garment_mask)

    # Shoulder-region corridors on each side, spanning from underarm 
    # up toward the collar, using the same y-range as the earlier 
    # single-sided test for direct comparability
    left_corridor = ((20, 140), (max(0, left_ua[0] - CORRIDOR_HALF_WIDTH), left_ua[0] + 40))
    right_corridor = ((20, 140), (right_ua[0] - 40, min(gw, right_ua[0] + CORRIDOR_HALF_WIDTH)))

    print(f"\nLeft corridor:  y={left_corridor[0]}  x={left_corridor[1]}")
    print(f"Right corridor: y={right_corridor[0]}  x={right_corridor[1]}")

    left_result = find_ridge_candidate(grad_mag, eroded_mask, *left_corridor)
    right_result = find_ridge_candidate(grad_mag, eroded_mask, *right_corridor)

    print(f"\n--- LEFT ridge candidate ---")
    print(left_result if left_result else "None found")
    print(f"\n--- RIGHT ridge candidate ---")
    print(right_result if right_result else "None found")

    if left_result and right_result:
        angle_diff = abs(left_result["angle_deg"] - right_result["angle_deg"])
        angle_diff = min(angle_diff, 180 - angle_diff)
        length_ratio = min(left_result["length"], right_result["length"]) / max(left_result["length"], right_result["length"])
        print(f"\n--- Bilateral comparison ---")
        print(f"  Angle difference: {angle_diff:.1f}°")
        print(f"  Length ratio (smaller/larger): {length_ratio:.2f}")

    # Visualize both corridors + detected components
    canvas = np.array(garment_img.convert("RGB")).copy()
    for (y_range, x_range), color in [(left_corridor, (0,255,0)), (right_corridor, (255,0,255))]:
        y0, y1 = y_range
        x0, x1 = x_range
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 1)

    scale = 3
    canvas_big = cv2.resize(canvas, (canvas.shape[1]*scale, canvas.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
    Image.fromarray(canvas_big).save(f"{OUTPUT_DIR}/internal_ridge_bilateral_test.png")
    print(f"\nSaved: {OUTPUT_DIR}/internal_ridge_bilateral_test.png (3x zoom, "
          f"green=left corridor, magenta=right corridor)")

    print(f"\n>>> DECISION GATE: do both sides show a real ridge candidate "
          f"with roughly corresponding length and angle (bilateral "
          f"evidence for a real seam), or is one side much weaker/absent "
          f"(evidence this is incidental, not a reliable landmark)?")


if __name__ == "__main__":
    main()