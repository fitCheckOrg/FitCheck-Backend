"""
First correspondence diagnostic. Extracts normalized width profiles
from BOTH representations independently, then visualizes them
side-by-side. No mapping, no deformation, no fitting decisions —
purely: do these two profiles have a plausible relationship?

Garment: alpha mask, normalized from underarm-level (not collar) 
to hem, since the pre-underarm region is sleeve-dominated and not 
comparable to the avatar's torso-only mask.
Avatar: SCHP torso mask, normalized top to bottom, as-is.

Run: python -m tryon.correspondence_profile_test
"""

import io
import os
import requests
import numpy as np
from PIL import Image

from tryon.avatar_representation import build_avatar_representation

GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
ALPHA_THRESHOLD = 10
OUTPUT_DIR = "outputs"
NUM_BINS = 20  # normalized vertical resolution for comparison


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD)


def extract_row_widths(mask):
    h, w = mask.shape
    rows = []
    for y in range(h):
        row = mask[y]
        xs = np.where(row)[0]
        if len(xs) == 0:
            continue
        rows.append((y, int(xs.min()), int(xs.max()), int(xs.max() - xs.min())))
    return rows


def normalize_profile(rows, y_start, y_end, num_bins):
    """
    Bins the width profile between y_start and y_end into num_bins
    equal-sized vertical segments, averaging width within each bin.
    Bins with no data are left as None — not interpolated.
    """
    span = y_end - y_start
    bins = [[] for _ in range(num_bins)]
    for y, left, right, width in rows:
        if y < y_start or y > y_end:
            continue
        bin_idx = min(int((y - y_start) / span * num_bins), num_bins - 1)
        bins[bin_idx].append(width)

    result = []
    for b in bins:
        result.append(np.mean(b) if b else None)
    return result


def print_profile(name, profile):
    print(f"\n{name} normalized width profile:")
    for i, w in enumerate(profile):
        frac = i / (len(profile) - 1)
        bar = "#" * int(w / 5) if w is not None else "(no data)"
        w_str = f"{w:.0f}px" if w is not None else "N/A"
        print(f"  {frac:.2f}  {w_str:>6}  {bar}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Fetching golden garment...")
    garment_resp = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{GOLDEN_GARMENT_ID}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    garment_bytes = requests.get(garment_resp["clean_image_url"]).content
    garment_mask = get_binary_mask(garment_bytes)
    garment_rows = extract_row_widths(garment_mask)

    # Garment: normalize from underarm level to hem. Using the 
    # already-confirmed underarm y (~222) as the torso-region start,
    # and the mask's own bottom as hem — both real, previously 
    # measured values, not new assumptions.
    garment_underarm_y = 222
    garment_hem_y = garment_rows[-1][0]
    garment_profile = normalize_profile(garment_rows, garment_underarm_y, garment_hem_y, NUM_BINS)

    print("Building AvatarRepresentation...")
    rep = build_avatar_representation()
    torso_rows = extract_row_widths(rep.torso_mask)
    torso_top_y = torso_rows[0][0]
    torso_bottom_y = torso_rows[-1][0]
    avatar_profile = normalize_profile(torso_rows, torso_top_y, torso_bottom_y, NUM_BINS)

    print_profile("GARMENT (underarm→hem)", garment_profile)
    print_profile("AVATAR (torso top→bottom)", avatar_profile)

    # Save raw data for direct inspection — no derived correspondence yet
    print(f"\n>>> DECISION GATE: do these two profiles show a plausible "
          f"relationship (e.g. both roughly steady, both taper the same "
          f"direction, comparable relative width changes) — or do they "
          f"diverge in ways that would make simple normalized-vertical "
          f"mapping invalid?")


if __name__ == "__main__":
    main()