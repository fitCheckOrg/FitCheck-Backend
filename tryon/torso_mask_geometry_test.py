"""
Torso mask geometry diagnostic. Queries the REAL torso mask
row-by-row for left/right boundary and width. No assumptions about
shape, no waist detection, no hem correspondence, no deformation —
purely: what does SCHP's torso mask actually look like, measured
directly.

Run: python -m tryon.torso_mask_geometry_test
"""

import os
import numpy as np
from PIL import Image, ImageDraw

from tryon.avatar_representation import build_avatar_representation

OUTPUT_DIR = "outputs"


def extract_row_boundaries(torso_mask):
    """
    For every row containing at least one torso pixel, find the
    leftmost and rightmost True pixel. Rows with no torso pixels
    are skipped entirely — not interpolated, not assumed.
    """
    h, w = torso_mask.shape
    rows = []

    for y in range(h):
        row = torso_mask[y]
        xs = np.where(row)[0]
        if len(xs) == 0:
            continue
        left = int(xs.min())
        right = int(xs.max())
        width = right - left
        rows.append((y, left, right, width))

    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Building AvatarRepresentation...")
    rep = build_avatar_representation()
    torso_mask = rep.torso_mask
    aw, ah = rep.image.size

    print(f"Image size: {aw} x {ah}")
    print(f"Torso mask total pixels: {torso_mask.sum()}")

    rows = extract_row_boundaries(torso_mask)

    if not rows:
        print("\n>>> NO TORSO PIXELS FOUND AT ALL. Representation failure.")
        return

    top_y = rows[0][0]
    bottom_y = rows[-1][0]
    widths = [r[3] for r in rows]
    max_width = max(widths)
    max_width_y = rows[widths.index(max_width)][0]
    min_width = min(widths)
    min_width_y = rows[widths.index(min_width)][0]

    print(f"\nTorso bounds:")
    print(f"  top:    y={top_y}")
    print(f"  bottom: y={bottom_y}")
    print(f"  height: {bottom_y - top_y}px")

    print(f"\nWidth profile (every 10th row):")
    for r in rows[::10]:
        y, left, right, width = r
        print(f"  y={y:4d}  left={left:4d}  right={right:4d}  width={width:4d}")

    print(f"\nMaximum width: {max_width}px at y={max_width_y}")
    print(f"Minimum width: {min_width}px at y={min_width_y}")

    # Check for discontinuities — rows where width changes drastically 
    # from the previous row, which might indicate mask noise rather 
    # than real anatomy. Reported, not corrected.
    print(f"\nLargest row-to-row width jumps (top 5, for inspection only):")
    jumps = []
    for i in range(1, len(rows)):
        prev_width = rows[i-1][3]
        curr_width = rows[i][3]
        jump = abs(curr_width - prev_width)
        jumps.append((jump, rows[i][0], prev_width, curr_width))
    jumps.sort(reverse=True)
    for jump, y, prev_w, curr_w in jumps[:5]:
        print(f"  y={y}: width jumped from {prev_w} to {curr_w} (Δ{jump})")

    # Visualization: original avatar with left/right boundary points 
    # drawn directly from the measured data, one dot per row (or 
    # every few rows for clarity)
    canvas = rep.image.copy()
    draw = ImageDraw.Draw(canvas)

    for r in rows[::3]:  # every 3rd row for visual density without clutter
        y, left, right, width = r
        draw.point((left, y), fill=(255, 0, 0))
        draw.point((right, y), fill=(0, 150, 255))

    # Mark top/bottom and max/min width rows explicitly
    for y, label, color in [
        (top_y, "TOP", (0, 255, 0)),
        (bottom_y, "BOTTOM", (0, 255, 0)),
        (max_width_y, "MAX_WIDTH", (255, 255, 0)),
        (min_width_y, "MIN_WIDTH", (255, 0, 255)),
    ]:
        draw.line([(0, y), (aw, y)], fill=color, width=1)
        draw.text((5, y), label, fill=color)

    canvas.save(f"{OUTPUT_DIR}/torso_mask_geometry_test.png")
    print(f"\nSaved: {OUTPUT_DIR}/torso_mask_geometry_test.png")
    print(f"\n>>> DECISION GATE: does the left/right boundary trace look like a "
          f"stable, anatomically sensible torso outline, or does it show "
          f"noise/discontinuities that indicate mask unreliability?")


if __name__ == "__main__":
    main()