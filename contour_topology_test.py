"""
v4 — Contour topology diagnostic. NOT a cuff detector yet.

Answers one architectural question: does walking the contour from
each underarm, in each direction, stay on the sleeve for a
meaningful distance before crossing into the torso? If yes, arc-
length-bounded traversal gives us a principled way to isolate the
sleeve that x/y filtering fundamentally cannot.

No scoring formula. No margin tuning. Just: print the path, and
draw it, so we can see with our own eyes which direction is which.

Run: python contour_topology_test.py
"""

import io
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw

ALPHA_THRESHOLD = 10
MIN_DEFECT_DEPTH = 1000
PRINT_INTERVAL = 20   # print a point roughly every 20px of arc length
PRINT_CAP = 260        # stop printing after this much arc length either way

TEST_CASES = [
    {"item_id": "25989827-db9d-4d65-90b3-12f5357b0b44", "label": "hanging_sleeve_shirt"},
    {"item_id": "ac7fffb7-2e3a-40f8-894a-0e85fc245373", "label": "polo_baseline"},
]

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255, image


def find_underarm_indices(main_contour, w_img):
    """Returns the actual CONTOUR INDEX for each underarm, not just
    the (x,y) point — required to walk the contour from there."""
    hull_indices = cv2.convexHull(main_contour, returnPoints=False)
    defects = cv2.convexityDefects(main_contour, hull_indices)
    defects_flat = defects.reshape(-1, 4)
    left_candidates, right_candidates = [], []
    for i in range(len(defects_flat)):
        _, _, far_idx, depth = defects_flat[i]
        depth, far_idx = int(depth), int(far_idx)
        if depth <= MIN_DEFECT_DEPTH:
            continue
        far_pt = main_contour[far_idx][0]
        x_pct = far_pt[0] / w_img * 100
        (left_candidates if x_pct < 50 else right_candidates).append((depth, far_idx))
    left_candidates.sort(reverse=True)
    right_candidates.sort(reverse=True)
    return left_candidates[0][1], right_candidates[0][1]


def contour_arc_distances(contour, start_idx):
    """Exact function as specified — cumulative contour distance
    in both directions from start_idx around the closed contour."""
    n = len(contour)
    forward = {}
    backward = {}

    dist = 0.0
    for step in range(n):
        idx = (start_idx + step) % n
        forward[idx] = dist
        next_idx = (idx + 1) % n
        p1 = contour[idx][0]
        p2 = contour[next_idx][0]
        dist += np.linalg.norm(p2.astype(float) - p1.astype(float))

    dist = 0.0
    for step in range(n):
        idx = (start_idx - step) % n
        backward[idx] = dist
        prev_idx = (idx - 1) % n
        p1 = contour[idx][0]
        p2 = contour[prev_idx][0]
        dist += np.linalg.norm(p2.astype(float) - p1.astype(float))

    return forward, backward


def trace_and_print(main_contour, start_idx, label):
    forward, backward = contour_arc_distances(main_contour, start_idx)

    print(f"\n{label} — start_idx={start_idx}, point={tuple(main_contour[start_idx][0])}")

    print(f"  FORWARD:")
    last_printed = -PRINT_INTERVAL
    for idx, d in forward.items():
        if d > PRINT_CAP:
            break
        if d - last_printed >= PRINT_INTERVAL:
            pt = tuple(int(v) for v in main_contour[idx][0])
            print(f"    +{d:.0f}px -> idx={idx} pos={pt}")
            last_printed = d

    print(f"  BACKWARD:")
    last_printed = -PRINT_INTERVAL
    for idx, d in backward.items():
        if d > PRINT_CAP:
            break
        if d - last_printed >= PRINT_INTERVAL:
            pt = tuple(int(v) for v in main_contour[idx][0])
            print(f"    +{d:.0f}px -> idx={idx} pos={pt}")
            last_printed = d

    return forward, backward


def visualize_traversal(image, main_contour, left_idx, right_idx,
                          left_fwd, left_bwd, right_fwd, right_bwd):
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    full_pts = [(int(p[0][0]), int(p[0][1])) for p in main_contour]
    draw.line(full_pts + [full_pts[0]], fill=(60, 60, 60), width=1)

    def draw_arc(arc_dict, color, cap):
        for idx, d in arc_dict.items():
            if d > cap:
                continue
            x, y = main_contour[idx][0]
            draw.point((int(x), int(y)), fill=color)

    # Left: forward=red, backward=orange
    draw_arc(left_fwd, (255, 0, 0), PRINT_CAP)
    draw_arc(left_bwd, (255, 165, 0), PRINT_CAP)
    # Right: forward=cyan, backward=magenta
    draw_arc(right_fwd, (0, 200, 255), PRINT_CAP)
    draw_arc(right_bwd, (255, 0, 255), PRINT_CAP)

    for idx, lbl in [(left_idx, "L_underarm"), (right_idx, "R_underarm")]:
        x, y = main_contour[idx][0]
        draw.ellipse([x-9, y-9, x+9, y+9], outline=(0, 255, 0), width=3)
        draw.text((x+11, y-8), lbl, fill=(0, 255, 0))

    return canvas


def main():
    for case in TEST_CASES:
        print(f"\n{'='*70}")
        print(f"{case['item_id']} — {case['label']}")
        print('='*70)

        result_data = requests.get(
            f"http://127.0.0.1:8000/api/wardrobe/{case['item_id']}?user_id={AVATAR_USER_ID}"
        ).json()["data"]
        image_bytes = requests.get(result_data["clean_image_url"]).content

        binary_mask, original = get_binary_mask(image_bytes)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_contour = max(contours, key=cv2.contourArea)
        w_img = binary_mask.shape[1]

        left_idx, right_idx = find_underarm_indices(main_contour, w_img)

        left_fwd, left_bwd = trace_and_print(main_contour, left_idx, "LEFT UNDERARM")
        right_fwd, right_bwd = trace_and_print(main_contour, right_idx, "RIGHT UNDERARM")

        canvas = visualize_traversal(original, main_contour, left_idx, right_idx,
                                        left_fwd, left_bwd, right_fwd, right_bwd)
        filename = f"topology_test_{case['label']}.png"
        canvas.save(filename)
        print(f"\nSaved: {filename}  "
              f"(L: forward=red, backward=orange | R: forward=cyan, backward=magenta)")


if __name__ == "__main__":
    main()