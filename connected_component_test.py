"""
Connected-component geometry experiment — tests whether the alpha
mask's topology can separate sleeve fabric from torso fabric, or
whether real garments form one connected silhouette (expected,
likely outcome for many shirts, per supervisor's own prediction).

This does NOT assume connected components will work. It's built
to clearly reveal which case we're in:
  A) Multiple distinct components exist, and they correspond 
     sensibly to GPT's horizontal sleeve/torso partition — real 
     signal to build on
  B) Everything is one connected blob — falsifies this hypothesis, 
     pointing toward contour/landmark-based geometry instead

Run: python connected_component_test.py
"""

import sys
import io
import requests
import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, ".")
from core.storage.supabase_client import supabase

TEST_CASES = [
    {
        "item_id": "25989827-db9d-4d65-90b3-12f5357b0b44",
        "label": "hanging-sleeve shirt",
        "left_sleeve_cols": (0, 20), "right_sleeve_cols": (80, 100), "torso_cols": (20, 80)
    },
    {
        "item_id": "ac7fffb7-2e3a-40f8-894a-0e85fc245373",
        "label": "extended-sleeve polo",
        "left_sleeve_cols": (0, 20), "right_sleeve_cols": (80, 100), "torso_cols": (20, 80)
    },
]

ALPHA_THRESHOLD = 10


def get_connected_components(image_bytes: bytes):
    """Labels every distinct connected region of opaque pixels in
    the FULL garment image (not restricted to any column range —
    that's the whole point, letting topology speak for itself)."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    binary_mask = alpha > ALPHA_THRESHOLD

    labeled_array, num_components = ndimage.label(binary_mask)
    return labeled_array, num_components, alpha.shape


def analyze_component_at_columns(labeled_array, h, w, left_pct, right_pct):
    """Which component label(s) actually appear within a given
    column range — reveals whether that region is its own distinct
    piece, or part of the same blob as everything else."""
    left_px = int(w * left_pct / 100)
    right_px = int(w * right_pct / 100)
    region = labeled_array[:, left_px:right_px]
    unique_labels = np.unique(region)
    unique_labels = unique_labels[unique_labels != 0]  # 0 = background
    return list(unique_labels)


def main():
    for case in TEST_CASES:
        print(f"\n{'='*60}")
        print(f"{case['item_id']} — {case['label']}")
        print('='*60)

        result = supabase.table("closet_items")\
            .select("clean_image_url").eq("item_id", case["item_id"]).single().execute()
        image_bytes = requests.get(result.data["clean_image_url"]).content

        labeled, num_components, (h, w) = get_connected_components(image_bytes)
        print(f"  Total distinct connected components found: {num_components}")

        if num_components == 1:
            print("  >>> HYPOTHESIS FALSIFIED for this garment: entire "
                  "silhouette is one connected blob. Sleeve and torso "
                  "are not topologically separable this way.")
            continue

        torso_labels = analyze_component_at_columns(labeled, h, w, *case["torso_cols"])
        left_labels = analyze_component_at_columns(labeled, h, w, *case["left_sleeve_cols"])
        right_labels = analyze_component_at_columns(labeled, h, w, *case["right_sleeve_cols"])

        print(f"  Component label(s) in torso columns: {torso_labels}")
        print(f"  Component label(s) in left sleeve columns: {left_labels}")
        print(f"  Component label(s) in right sleeve columns: {right_labels}")

        torso_set = set(torso_labels)
        left_set = set(left_labels)
        right_set = set(right_labels)

        if torso_set and left_set and torso_set.isdisjoint(left_set):
            print("  >>> Left sleeve region uses a DIFFERENT component than torso — real separation")
        elif torso_set & left_set:
            print("  >>> Left sleeve region SHARES a component with torso — no real separation")

        if torso_set and right_set and torso_set.isdisjoint(right_set):
            print("  >>> Right sleeve region uses a DIFFERENT component than torso — real separation")
        elif torso_set & right_set:
            print("  >>> Right sleeve region SHARES a component with torso — no real separation")


if __name__ == "__main__":
    main()