"""
Semantic avatar anchor diagnostic. Derives underarm candidates from
the actual boundary between torso_mask and upper_arms_mask — real
semantic adjacency, not contour-shape inference. Shoulders and hips
kept as separate MediaPipe observations, NOT used to define any
garment destination yet.

No correspondence, no deformation, no fitting decisions.

Run: python -m tryon.avatar_torso_landmarks_test
"""

import os
import numpy as np
from PIL import Image, ImageDraw

from tryon.avatar_representation import build_avatar_representation

OUTPUT_DIR = "outputs"
ADJACENCY_GAP_PX = 5  # max horizontal gap to count torso/arm as "touching"


def find_semantic_underarm(torso_mask, upper_arms_mask, image_width, side):
    """
    For the given side ('left' or 'right', in IMAGE space — smaller
    x = left half of image), scans rows top to bottom. A row counts
    as "arm attached to torso" if both torso and upper_arm pixels
    exist on that side AND the arm's inner boundary sits within
    ADJACENCY_GAP_PX of the torso's outer boundary on that side.

    Returns the LAST such row (scanning downward) — i.e. where the
    arm stops being adjacent to the torso — as the semantic underarm
    y-coordinate. Returns None if no adjacency is ever found.
    """
    h, w = torso_mask.shape
    mid_x = image_width // 2
    last_adjacent_y = None
    adjacent_rows = []

    for y in range(h):
        torso_row = torso_mask[y]
        arm_row = upper_arms_mask[y]

        if side == "left":
            torso_xs = np.where(torso_row[:mid_x])[0]
            arm_xs = np.where(arm_row[:mid_x])[0]
        else:
            torso_xs = np.where(torso_row[mid_x:])[0] + mid_x
            arm_xs = np.where(arm_row[mid_x:])[0] + mid_x

        if len(torso_xs) == 0 or len(arm_xs) == 0:
            continue

        if side == "left":
            torso_outer = torso_xs.min()  # torso's outer edge on the left side
            arm_inner = arm_xs.max()      # arm's inner edge (closest to torso)
            gap = torso_outer - arm_inner
        else:
            torso_outer = torso_xs.max()
            arm_inner = arm_xs.min()
            gap = arm_inner - torso_outer

        if -ADJACENCY_GAP_PX <= gap <= ADJACENCY_GAP_PX:
            adjacent_rows.append(y)
            last_adjacent_y = y

    return last_adjacent_y, adjacent_rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Building AvatarRepresentation...")
    rep = build_avatar_representation()
    aw, ah = rep.image.size

    print(f"Image size: {aw} x {ah}")

    left_underarm_y, left_adjacent_rows = find_semantic_underarm(
        rep.torso_mask, rep.upper_arms_mask, aw, "left"
    )
    right_underarm_y, right_adjacent_rows = find_semantic_underarm(
        rep.torso_mask, rep.upper_arms_mask, aw, "right"
    )

    print(f"\nLeft side (image-space):  {len(left_adjacent_rows)} adjacent rows found")
    print(f"  Semantic underarm y: {left_underarm_y}")
    print(f"Right side (image-space): {len(right_adjacent_rows)} adjacent rows found")
    print(f"  Semantic underarm y: {right_underarm_y}")

    # Get actual x-coordinate of torso boundary at the underarm rows
    left_underarm_pt = None
    right_underarm_pt = None
    if left_underarm_y is not None:
        row = rep.torso_mask[left_underarm_y]
        xs = np.where(row[:aw//2])[0]
        left_underarm_pt = (int(xs.min()), left_underarm_y)
    if right_underarm_y is not None:
        row = rep.torso_mask[right_underarm_y]
        xs = np.where(row[aw//2:])[0] + aw // 2
        right_underarm_pt = (int(xs.max()), right_underarm_y)

    print(f"\nSemantic underarm points:")
    print(f"  Left:  {left_underarm_pt}")
    print(f"  Right: {right_underarm_pt}")

    print(f"\nMediaPipe shoulders (observation only):")
    print(f"  Left:  {rep.left_shoulder}")
    print(f"  Right: {rep.right_shoulder}")
    print(f"\nMediaPipe hips (observation only, NOT a garment destination):")
    print(f"  Left:  {rep.left_hip}")
    print(f"  Right: {rep.right_hip}")

    # Visualization
    canvas = rep.image.copy()
    overlay = np.array(canvas).astype(np.int32)

    torso_tint = np.array([0, 100, 0])
    arm_tint = np.array([100, 0, 0])
    torso_bool = np.stack([rep.torso_mask]*3, axis=-1)
    arm_bool = np.stack([rep.upper_arms_mask]*3, axis=-1)
    overlay = np.where(torso_bool, (overlay + torso_tint), overlay)
    overlay = np.where(arm_bool, (overlay + arm_tint), overlay)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(overlay)

    draw = ImageDraw.Draw(canvas)

    for pt, color, label in [
        (rep.left_shoulder, (0, 255, 0), "L_shoulder"),
        (rep.right_shoulder, (0, 255, 0), "R_shoulder"),
        (rep.left_hip, (255, 165, 0), "L_hip"),
        (rep.right_hip, (255, 165, 0), "R_hip"),
    ]:
        x, y = pt
        draw.ellipse([x-5, y-5, x+5, y+5], outline=color, width=2)
        draw.text((x+7, y-5), label, fill=color)

    for pt, color, label in [
        (left_underarm_pt, (255, 0, 255), "L_underarm(semantic)"),
        (right_underarm_pt, (0, 200, 255), "R_underarm(semantic)"),
    ]:
        if pt is None:
            continue
        x, y = pt
        draw.ellipse([x-7, y-7, x+7, y+7], outline=color, width=3)
        draw.text((x+9, y-7), label, fill=color)

    canvas.save(f"{OUTPUT_DIR}/avatar_torso_landmarks_test.png")
    print(f"\nSaved: {OUTPUT_DIR}/avatar_torso_landmarks_test.png")
    print(f"\n>>> DECISION GATE: do the semantic underarm points (magenta/cyan) "
          f"land at anatomically sensible armpit locations, distinct from and "
          f"more reliable than shoulder or hip landmarks alone?")


if __name__ == "__main__":
    main()