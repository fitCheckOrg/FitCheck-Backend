"""tryon/diagnose_underarm_components.py - read-only, no API credits"""
import os
import cv2
import numpy as np
from PIL import Image
from tryon.avatar_representation import build_avatar_representation

NEW_USER_ID = "f4b5f129-db34-40e5-8423-d1b256a15578"
SCALE = 4


def components(mask, name):
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    print(f"{name}: {n - 1} blob(s)")
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 20:
            continue
        print(f"   x={x}..{x + bw - 1}  y={y}..{y + bh - 1}  area={area}  centroid_x={cents[i][0]:.0f}")


def main():
    rep = build_avatar_representation(avatar_user_id=NEW_USER_ID)
    w, h = rep.image.size
    torso, arms = rep.torso_mask, rep.upper_arms_mask
    xs = np.where(torso.any(axis=0))[0]
    print(f"image {w}x{h}   image midline x={w // 2}")
    print(f"torso x-range {xs.min()}..{xs.max()}   torso center x={(xs.min() + xs.max()) / 2:.0f}\n")
    components(torso, "torso")
    components(arms, "upper_arms")

    canvas = np.array(rep.image.convert("RGB")).astype(np.float32)
    canvas[torso] = canvas[torso] * 0.5 + np.array([0, 255, 0]) * 0.5
    canvas[arms] = canvas[arms] * 0.5 + np.array([255, 0, 0]) * 0.5
    canvas[:, w // 2] = [255, 255, 0]
    img = Image.fromarray(canvas.astype(np.uint8)).resize((w * SCALE, h * SCALE), Image.NEAREST)
    os.makedirs("outputs", exist_ok=True)
    img.save("outputs/new_avatar_masks_4x.png")
    print("\nSaved outputs/new_avatar_masks_4x.png (green=torso, red=upper arms, yellow=image midline)")


if __name__ == "__main__":
    main()