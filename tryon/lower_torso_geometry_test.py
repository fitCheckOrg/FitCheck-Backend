"""
Diagnostic: what does the SCHP torso mask's own lower boundary
actually look like? No assumption that "bottom of torso mask" =
hip or waist. Pure measurement of real semantic pixels.

Investigates:
1. Torso mask bottom boundary shape (does it end cleanly, or 
   fade/fragment?)
2. Left/right boundary behavior in the lower third
3. Width profile through the lower torso specifically
4. Whether a stable bilateral feature exists near the bottom
5. Whether the boundary is distinguishable from the underwear/
   shorts edge, or confounded with it (SCHP labels clothing 
   separately from skin in its broader model, but THIS torso 
   class is skin/body region only — worth confirming directly, 
   not assuming)

Run: python -m tryon.lower_torso_geometry_test
"""

import os
import numpy as np
from PIL import Image, ImageDraw

from tryon.avatar_representation import build_avatar_representation

OUTPUT_DIR = "outputs"


def extract_row_boundaries(mask):
    h, w = mask.shape
    rows = []
    for y in range(h):
        row = mask[y]
        xs = np.where(row)[0]
        if len(xs) == 0:
            continue
        rows.append((y, int(xs.min()), int(xs.max()), int(xs.max() - xs.min())))
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Building AvatarRepresentation...")
    rep = build_avatar_representation()
    aw, ah = rep.image.size
    torso_mask = rep.torso_mask

    rows = extract_row_boundaries(torso_mask)
    top_y = rows[0][0]
    bottom_y = rows[-1][0]
    total_height = bottom_y - top_y

    print(f"\nTorso mask: top y={top_y}  bottom y={bottom_y}  height={total_height}px")

    # Focus on the LOWER THIRD specifically
    lower_third_start = top_y + int(total_height * 0.67)
    lower_rows = [r for r in rows if r[0] >= lower_third_start]

    print(f"\n=== LOWER THIRD (y={lower_third_start} to y={bottom_y}) ===")
    print(f"{'y':<6}{'left':<6}{'right':<6}{'width':<6}")
    for y, left, right, width in lower_rows[::3]:
        print(f"{y:<6}{left:<6}{right:<6}{width:<6}")

    # Check the last several rows specifically — does the mask end 
    # cleanly (sharp cutoff) or fade/fragment (width shrinking 
    # erratically, or many small disconnected segments)?
    print(f"\n=== FINAL 15 ROWS (raw, unsampled) — cleanly ending or fragmenting? ===")
    for y, left, right, width in rows[-15:]:
        print(f"  y={y}  left={left}  right={right}  width={width}")

    # Width trend through lower third: smooth taper, or a distinct 
    # bilateral narrowing (a real "waist-like" feature) vs shorts-edge 
    # confound (would show as an abrupt, non-anatomical jump)
    widths = [r[3] for r in lower_rows]
    print(f"\nLower-third width range: {min(widths)}-{max(widths)}px")
    diffs = [abs(widths[i]-widths[i-1]) for i in range(1, len(widths))]
    if diffs:
        print(f"Largest single-step width change in lower third: {max(diffs)}px "
              f"(large jumps here would suggest confound with clothing edge, "
              f"not real anatomy)")

    # Visualize: torso mask boundary + explicit marker at bottom_y, 
    # for direct inspection against the actual photo
    canvas = rep.image.copy()
    draw = ImageDraw.Draw(canvas)
    for y, left, right, width in rows[::3]:
        draw.point((left, y), fill=(255, 0, 0))
        draw.point((right, y), fill=(0, 150, 255))
    draw.line([(0, lower_third_start), (aw, lower_third_start)], fill=(255,255,0), width=1)
    draw.text((5, lower_third_start), "lower-third start", fill=(255,255,0))
    draw.line([(0, bottom_y), (aw, bottom_y)], fill=(0,255,0), width=1)
    draw.text((5, bottom_y), "torso mask bottom", fill=(0,255,0))

    canvas.save(f"{OUTPUT_DIR}/lower_torso_geometry_test.png")
    print(f"\nSaved: {OUTPUT_DIR}/lower_torso_geometry_test.png")
    print(f"\n>>> DECISION GATE: does the torso mask end with a stable, "
          f"anatomically plausible bilateral boundary (real waist-like "
          f"narrowing), or does it fragment/end abruptly in a way that "
          f"suggests it's just tracking the visible skin-to-underwear "
          f"edge rather than real torso anatomy?")


if __name__ == "__main__":
    main()