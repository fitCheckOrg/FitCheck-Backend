"""
Tests whether a full affine transform (fit via least squares across
BOTH underarm and shoulder correspondence pairs) satisfies all four
anchor points better than the underarm-only similarity transform.

Affine = 6 degrees of freedom (vs similarity's 4: scale, rotation,
2D translation). Computed directly from the 4 known point pairs,
not assumed.

Run: python -m tryon.correspondence_affine_test
"""

import io
import os
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw

from tryon.avatar_representation import build_avatar_representation
from tryon.avatar_torso_landmarks_test import find_semantic_underarm
from tryon.correspondence_shoulders_test import find_garment_underarms, find_garment_shoulder_points

GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
ALPHA_THRESHOLD = 10
OUTPUT_DIR = "outputs"


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD), image


def fit_affine_least_squares(src_points, dst_points):
    """
    Fits the best affine transform (6 DOF: a,b,c,d,e,f in
    x' = a*x + b*y + e
    y' = c*x + d*y + f
    ) minimizing squared error across ALL given point pairs —
    not just 3 exact points, since we have 4 pairs (over-determined).
    """
    src = np.array(src_points, dtype=np.float64)
    dst = np.array(dst_points, dtype=np.float64)
    n = len(src)

    A = np.zeros((2 * n, 6))
    b = np.zeros(2 * n)
    for i in range(n):
        x, y = src[i]
        A[2*i]   = [x, y, 0, 0, 1, 0]
        A[2*i+1] = [0, 0, x, y, 0, 1]
        b[2*i]   = dst[i][0]
        b[2*i+1] = dst[i][1]

    params, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    a, bb, c, d, e, f = params
    matrix = np.array([[a, bb, e], [c, d, f]])
    return matrix


def apply_affine_to_point(matrix, pt):
    x, y = pt
    new_x = matrix[0,0]*x + matrix[0,1]*y + matrix[0,2]
    new_y = matrix[1,0]*x + matrix[1,1]*y + matrix[1,2]
    return (new_x, new_y)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Fetching golden garment...")
    garment_resp = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{GOLDEN_GARMENT_ID}?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    garment_bytes = requests.get(garment_resp["clean_image_url"]).content
    garment_mask, garment_img = get_binary_mask(garment_bytes)
    gw, gh = garment_img.size

    g_left_ua, g_right_ua = find_garment_underarms(garment_mask, gw)
    g_left_shoulder, g_right_shoulder = find_garment_shoulder_points(garment_mask, gw)
    print(f"Garment underarms: L={g_left_ua}  R={g_right_ua}")
    print(f"Garment shoulders: L={g_left_shoulder}  R={g_right_shoulder}")

    print("\nBuilding AvatarRepresentation...")
    rep = build_avatar_representation()
    aw, ah = rep.image.size

    a_left_underarm_y, _ = find_semantic_underarm(rep.torso_mask, rep.upper_arms_mask, aw, "left")
    a_right_underarm_y, _ = find_semantic_underarm(rep.torso_mask, rep.upper_arms_mask, aw, "right")
    a_left_row = rep.torso_mask[a_left_underarm_y]
    a_left_xs = np.where(a_left_row[:aw//2])[0]
    a_left_ua = (int(a_left_xs.min()), a_left_underarm_y)
    a_right_row = rep.torso_mask[a_right_underarm_y]
    a_right_xs = np.where(a_right_row[aw//2:])[0] + aw // 2
    a_right_ua = (int(a_right_xs.max()), a_right_underarm_y)

    print(f"Avatar semantic underarms: L={a_left_ua}  R={a_right_ua}")
    print(f"Avatar MediaPipe shoulders: L={rep.left_shoulder}  R={rep.right_shoulder}")

    # Pair by physical x-position (small-x <-> small-x), not label name
    g_small_ua = g_left_ua if g_left_ua[0] < g_right_ua[0] else g_right_ua
    g_large_ua = g_right_ua if g_left_ua[0] < g_right_ua[0] else g_left_ua
    g_small_sh = g_left_shoulder if g_left_shoulder[0] < g_right_shoulder[0] else g_right_shoulder
    g_large_sh = g_right_shoulder if g_left_shoulder[0] < g_right_shoulder[0] else g_left_shoulder

    a_small_ua = a_left_ua if a_left_ua[0] < a_right_ua[0] else a_right_ua
    a_large_ua = a_right_ua if a_left_ua[0] < a_right_ua[0] else a_left_ua
    a_small_sh = rep.left_shoulder if rep.left_shoulder[0] < rep.right_shoulder[0] else rep.right_shoulder
    a_large_sh = rep.right_shoulder if rep.left_shoulder[0] < rep.right_shoulder[0] else rep.left_shoulder

    src_points = [g_small_ua, g_large_ua, g_small_sh, g_large_sh]
    dst_points = [a_small_ua, a_large_ua, a_small_sh, a_large_sh]

    matrix = fit_affine_least_squares(src_points, dst_points)
    print(f"\nFitted affine matrix:\n{matrix}")

    print(f"\nResidual error at each anchor (should be small if affine is sufficient):")
    total_error = 0
    for name, src, dst in zip(
        ["L_underarm", "R_underarm", "L_shoulder", "R_shoulder"],
        src_points, dst_points
    ):
        transformed = apply_affine_to_point(matrix, src)
        error = np.hypot(transformed[0]-dst[0], transformed[1]-dst[1])
        total_error += error
        print(f"  {name}: transformed={tuple(round(v,1) for v in transformed)}  "
              f"target={dst}  error={error:.1f}px")
    print(f"  Total residual error: {total_error:.1f}px  "
          f"(compare to similarity transform's ~61px combined shoulder error)")

    # Warp the garment image using the fitted affine matrix
    warped = cv2.warpAffine(
        np.array(garment_img), matrix, (aw, ah),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT
    )
    warped_img = Image.fromarray(warped)

    result = rep.image.convert("RGBA").copy()
    result.paste(warped_img, (0, 0), warped_img)

    draw = ImageDraw.Draw(result)
    for name, src, dst in zip(
        ["L_underarm", "R_underarm", "L_shoulder", "R_shoulder"],
        src_points, dst_points
    ):
        transformed = apply_affine_to_point(matrix, src)
        x, y = transformed
        draw.ellipse([x-5, y-5, x+5, y+5], outline=(255, 0, 255), width=2)
        dx, dy = dst
        draw.ellipse([dx-5, dy-5, dx+5, dy+5], outline=(0, 255, 0), width=2)

    result.convert("RGB").save(f"{OUTPUT_DIR}/correspondence_affine_test.png")
    print(f"\nSaved: {OUTPUT_DIR}/correspondence_affine_test.png")
    print(f"\n>>> DECISION GATE: does the affine fit bring all 4 anchors close "
          f"together (magenta near green), or does satisfying shoulders now "
          f"break underarms — revealing that even affine isn't enough and "
          f"a piecewise/mesh approach is required?")


if __name__ == "__main__":
    main()