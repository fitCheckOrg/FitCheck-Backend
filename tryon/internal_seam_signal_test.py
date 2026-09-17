"""
Tests whether an internal seam/construction line is visible near the
expected shoulder region, independent of the outer contour. Scans a
band INSIDE the garment silhouette (not at the edge) between the
neckline and the outer sleeve corner already found, looking for a
locally consistent edge/gradient ridge that could correspond to a
physical seam.

Pure diagnostic — no landmark extraction, no promotion into the
pipeline. Visualizes raw gradient magnitude in the search region so
we can inspect whether ANY internal structure exists there at all.

Run: python -m tryon.internal_seam_signal_test
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

# Search region bounds, from already-established landmarks:
# outer sleeve corner ~(485,126), collar/neck region ~(305,24)
SEARCH_Y_RANGE = (20, 140)
SEARCH_X_RANGE = (300, 460)


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
    print(f"Search region: y={SEARCH_Y_RANGE}, x={SEARCH_X_RANGE}")

    # Compute gradient magnitude (Sobel) across the whole search region
    y0, y1 = SEARCH_Y_RANGE
    x0, x1 = SEARCH_X_RANGE
    region_gray = gray[y0:y1, x0:x1]
    region_mask = garment_mask[y0:y1, x0:x1]

    gx = cv2.Sobel(region_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(region_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mag[~region_mask] = 0  # only inside the garment

    print(f"\nGradient magnitude stats in search region (inside garment only):")
    inside_vals = grad_mag[region_mask]
    if len(inside_vals) > 0:
        print(f"  mean={inside_vals.mean():.2f}  max={inside_vals.max():.2f}  "
              f"std={inside_vals.std():.2f}  90th_pct={np.percentile(inside_vals, 90):.2f}")

    # Find the strongest connected edge structure via simple thresholding
    threshold = np.percentile(inside_vals, 95) if len(inside_vals) > 0 else 0
    strong_edges = (grad_mag > threshold) & region_mask
    print(f"  Pixels above 95th percentile ({threshold:.1f}): {strong_edges.sum()}")

    # Visualize: original crop + gradient magnitude heatmap side by side
    crop_orig = np.array(garment_img.convert("RGB"))[y0:y1, x0:x1]
    grad_norm = (grad_mag / (grad_mag.max() + 1e-6) * 255).astype(np.uint8)
    grad_color = cv2.applyColorMap(grad_norm, cv2.COLORMAP_JET)
    grad_color = cv2.cvtColor(grad_color, cv2.COLOR_BGR2RGB)
    grad_color[~region_mask] = [0, 0, 0]

    scale = 4
    crop_resized = cv2.resize(crop_orig, (crop_orig.shape[1]*scale, crop_orig.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
    grad_resized = cv2.resize(grad_color, (grad_color.shape[1]*scale, grad_color.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
    combined = np.concatenate([crop_resized, grad_resized], axis=1)

    Image.fromarray(combined).save(f"{OUTPUT_DIR}/internal_seam_signal_test.png")
    print(f"\nSaved: {OUTPUT_DIR}/internal_seam_signal_test.png (4x zoom, "
          f"original | gradient heatmap side by side)")
    print(f"\n>>> Look for: does the gradient heatmap show a coherent LINE "
          f"or RIDGE running through the region (a real seam), or just "
          f"scattered/uniform low-level noise (no internal signal to use)?")


if __name__ == "__main__":
    main()