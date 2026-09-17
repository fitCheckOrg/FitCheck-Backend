"""
Searches for the garment's real sleeve/shoulder structural junction
as a directional-trend transition along the contour between the
underarm and the collar — not a curvature-peak search, since the
validation image showed this is a gradual boundary change, not a
sharp corner.

For each side: walk the contour from the underarm toward the top,
tracking the local direction vector. Report where that direction
shifts from "mostly vertical" (sleeve outer edge) to "mostly
horizontal" (shoulder top) — the transition zone itself, since this
may not be a single point.

Pure diagnostic. No correspondence, no deformation.

Run: python -m tryon.garment_shoulder_junction_test
"""

import io
import os
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw

from tryon.correspondence_shoulders_test import find_garment_underarms

GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
ALPHA_THRESHOLD = 10
DIRECTION_WINDOW = 6
OUTPUT_DIR = "outputs"


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD), image


def find_nearest_contour_index(contour, target):
    tx, ty = target
    best_idx, best_dist = None, float('inf')
    for i, pt in enumerate(contour):
        x, y = pt[0]
        d = np.hypot(x - tx, y - ty)
        if d < best_dist:
            best_dist, best_idx = d, i
    return best_idx


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


def walk_toward_collar(contour, underarm_idx, arc_dict, max_arc, direction_window=DIRECTION_WINDOW):
    """
    Walks along the given arc direction from the underarm, up to
    max_arc, computing local direction angle (relative to vertical)
    at each sampled point. Returns the list of (arc_dist, point, angle_from_vertical).
    """
    ordered = sorted([(d, idx) for idx, d in arc_dict.items() if d <= max_arc], key=lambda x: x[0])
    indices = [idx for d, idx in ordered]

    results = []
    for i in range(direction_window, len(indices) - direction_window):
        idx = indices[i]
        prev_idx = indices[i - direction_window]
        next_idx = indices[i + direction_window]
        p_prev = contour[prev_idx][0].astype(float)
        p_next = contour[next_idx][0].astype(float)
        direction = p_next - p_prev
        angle_from_vertical = np.degrees(np.arctan2(abs(direction[0]), abs(direction[1])))
        # 0 degrees = perfectly vertical (sleeve edge), 90 = perfectly horizontal (shoulder top)
        pt = tuple(int(v) for v in contour[idx][0])
        arc_d = ordered[i][0]
        results.append((arc_d, pt, angle_from_vertical))

    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Fetching golden garment...")
    garment_resp = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{GOLDEN_GARMENT_ID}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    garment_bytes = requests.get(garment_resp["clean_image_url"]).content
    garment_mask, garment_img = get_binary_mask(garment_bytes)
    gw, gh = garment_img.size

    contours, _ = cv2.findContours(garment_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)

    left_ua, right_ua = find_garment_underarms(garment_mask, gw)
    print(f"Underarms: L={left_ua}  R={right_ua}")

    left_idx = find_nearest_contour_index(main_contour, left_ua)
    right_idx = find_nearest_contour_index(main_contour, right_ua)

    left_fwd, left_bwd = contour_arc_distances(main_contour, left_idx)
    right_fwd, right_bwd = contour_arc_distances(main_contour, right_idx)

    # Walk toward the collar — try both directions, use whichever 
    # reaches a shorter arc within a reasonable max (collar is 
    # "nearby" compared to going all the way around)
    max_walk = 200  # generous upper bound, garment-scale appropriate

    print(f"\n--- LEFT side: walking from underarm toward collar ---")
    left_walk_fwd = walk_toward_collar(main_contour, left_idx, left_fwd, max_walk)
    left_walk_bwd = walk_toward_collar(main_contour, left_idx, left_bwd, max_walk)
    left_walk = left_walk_fwd if len(left_walk_fwd) > len(left_walk_bwd) else left_walk_bwd
    for arc_d, pt, angle in left_walk[::4]:
        print(f"  arc={arc_d:6.1f}  pt={pt}  angle_from_vertical={angle:5.1f}°")

    print(f"\n--- RIGHT side: walking from underarm toward collar ---")
    right_walk_fwd = walk_toward_collar(main_contour, right_idx, right_fwd, max_walk)
    right_walk_bwd = walk_toward_collar(main_contour, right_idx, right_bwd, max_walk)
    right_walk = right_walk_fwd if len(right_walk_fwd) > len(right_walk_bwd) else right_walk_bwd
    for arc_d, pt, angle in right_walk[::4]:
        print(f"  arc={arc_d:6.1f}  pt={pt}  angle_from_vertical={angle:5.1f}°")

    # Find the transition point: where angle crosses 45 degrees 
    # (roughly halfway between "vertical sleeve edge" and "horizontal 
    # shoulder top"), reported as a candidate junction zone, not a 
    # single definitive point
    def find_transition(walk):
        for i in range(len(walk) - 1):
            if walk[i][2] < 45 <= walk[i+1][2]:
                return walk[i+1]
        return None

    left_transition = find_transition(left_walk)
    right_transition = find_transition(right_walk)

    print(f"\nCandidate LEFT junction (angle crosses 45°): {left_transition}")
    print(f"Candidate RIGHT junction (angle crosses 45°): {right_transition}")

    # Visualization
    canvas = garment_img.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    pts = [(int(p[0][0]), int(p[0][1])) for p in main_contour]
    draw.line(pts + [pts[0]], fill=(150, 150, 150), width=1)

    for walk, color in [(left_walk, (255, 0, 0)), (right_walk, (0, 150, 255))]:
        for arc_d, pt, angle in walk:
            x, y = pt
            draw.point((x, y), fill=color)

    for pt, color, label in [(left_ua, (0,255,0), "L_UA"), (right_ua, (0,255,0), "R_UA")]:
        x, y = pt
        draw.ellipse([x-6,y-6,x+6,y+6], outline=color, width=3)
        draw.text((x+8,y-6), label, fill=color)

    for transition, color, label in [
        (left_transition, (255,255,0), "L_junction_candidate"),
        (right_transition, (255,0,255), "R_junction_candidate"),
    ]:
        if transition is None:
            continue
        _, pt, _ = transition
        x, y = pt
        draw.ellipse([x-8,y-8,x+8,y+8], outline=color, width=3)
        draw.text((x+10,y-8), label, fill=color)

    canvas.save(f"{OUTPUT_DIR}/garment_shoulder_junction_test.png")
    print(f"\nSaved: {OUTPUT_DIR}/garment_shoulder_junction_test.png")
    print(f"\n>>> DECISION GATE: does the candidate junction point (yellow/"
          f"magenta) land visually at the real sleeve-to-shoulder "
          f"structural transition, closer to what the collar-proxy check "
          f"revealed as the true location?")


if __name__ == "__main__":
    main()