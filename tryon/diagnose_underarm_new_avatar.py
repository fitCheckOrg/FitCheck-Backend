"""tryon/diagnose_underarm_new_avatar.py - read-only, no API credits"""
import os
import numpy as np
from PIL import Image
from tryon.avatar_representation import build_avatar_representation
from tryon.avatar_torso_landmarks_test import find_semantic_underarm, ADJACENCY_GAP_PX

NEW_USER_ID = "f4b5f129-db34-40e5-8423-d1b256a15578"


def side_gaps(torso, arms, w, side):
    mid = w // 2
    out = []
    for y in range(torso.shape[0]):
        if side == "left":
            t = np.where(torso[y, :mid])[0]
            a = np.where(arms[y, :mid])[0]
            if len(t) and len(a):
                out.append((y, int(t.min() - a.max())))
        else:
            t = np.where(torso[y, mid:])[0] + mid
            a = np.where(arms[y, mid:])[0] + mid
            if len(t) and len(a):
                out.append((y, int(a.min() - t.max())))
    return out


def main():
    rep = build_avatar_representation(avatar_user_id=NEW_USER_ID)
    w, h = rep.image.size
    torso, arms = rep.torso_mask, rep.upper_arms_mask
    print(f"image {w}x{h}  torso px={int(torso.sum())}  upper_arms px={int(arms.sum())}")
    print(f"adjacency threshold = {ADJACENCY_GAP_PX}px\n")

    for side in ("left", "right"):
        y, rows = find_semantic_underarm(torso, arms, w, side)
        gaps = side_gaps(torso, arms, w, side)
        print(f"[{side}] returned y={y}  adjacent rows={len(rows)}  rows with torso+arm pixels={len(gaps)}")
        if gaps:
            print(f"   closest rows (y, gap_px): {sorted(gaps, key=lambda g: abs(g[1]))[:5]}")
        else:
            print("   no row has both torso and upper-arm pixels on this side")
        print()

    os.makedirs("outputs", exist_ok=True)
    canvas = np.array(rep.image.convert("RGB")).astype(np.float32)
    canvas[torso] = canvas[torso] * 0.5 + np.array([0, 255, 0]) * 0.5
    canvas[arms] = canvas[arms] * 0.5 + np.array([255, 0, 0]) * 0.5
    canvas[:, w // 2] = [255, 255, 0]
    Image.fromarray(canvas.astype(np.uint8)).save("outputs/new_avatar_masks.png")
    print("Saved outputs/new_avatar_masks.png (green=torso, red=upper arms, yellow=midline)")


if __name__ == "__main__":
    main()