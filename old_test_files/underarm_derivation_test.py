"""
Avatar-side underarm derivation test — narrowed range.

Tests U(t) = S + t(H - S) for t in [0.08, 0.10, 0.12, 0.14, 0.16, 0.18],
against REAL avatar image + REAL stored MediaPipe pose_keypoints.

This range is narrowed based on the FIRST photo's evidence (0.05 too 
high, 0.25-0.30 too low) — not an assumption the answer lives here, 
just spending resolution where the first test's evidence pointed.

Run against 2-3 genuinely different avatar photos (different poses/
bodies), record best-t and plausible-range for EACH separately.
Do not average or pick a winner until all photos are compared.

Run: python underarm_derivation_test.py
"""

import io
import requests
from PIL import Image, ImageDraw

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
T_VALUES = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18]


def fetch_avatar():
    resp = requests.get(
        f"http://127.0.0.1:8000/api/avatar/me?user_id={AVATAR_USER_ID}"
    ).json()
    data = resp["data"]
    img_resp = requests.get(data["processed_photo_url"])
    avatar_img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
    return avatar_img, data["pose_keypoints"]


def to_pixel(point_pct, w, h):
    return (point_pct["x"] * w, point_pct["y"] * h)


def interpolate(shoulder_px, hip_px, t):
    x = shoulder_px[0] + t * (hip_px[0] - shoulder_px[0])
    y = shoulder_px[1] + t * (hip_px[1] - shoulder_px[1])
    return (x, y)


def main():
    print("Fetching real avatar and pose_keypoints...")
    avatar_img, keypoints = fetch_avatar()
    w, h = avatar_img.size

    left_shoulder = to_pixel(keypoints["left_shoulder"], w, h)
    right_shoulder = to_pixel(keypoints["right_shoulder"], w, h)
    left_hip = to_pixel(keypoints["left_hip"], w, h)
    right_hip = to_pixel(keypoints["right_hip"], w, h)

    print(f"\nleft_shoulder:  ({left_shoulder[0]:.1f}, {left_shoulder[1]:.1f})")
    print(f"right_shoulder: ({right_shoulder[0]:.1f}, {right_shoulder[1]:.1f})")
    print(f"left_hip:       ({left_hip[0]:.1f}, {left_hip[1]:.1f})")
    print(f"right_hip:      ({right_hip[0]:.1f}, {right_hip[1]:.1f})")

    canvas = avatar_img.copy()
    draw = ImageDraw.Draw(canvas)

    def mark(pt, color, radius, label=None):
        x, y = pt
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], outline=color, width=2)
        if label:
            draw.text((x+radius+3, y-6), label, fill=color)

    mark(left_shoulder, (0, 255, 0), 8, "L_shoulder")
    mark(right_shoulder, (0, 255, 0), 8, "R_shoulder")
    mark(left_hip, (255, 165, 0), 8, "L_hip")
    mark(right_hip, (255, 165, 0), 8, "R_hip")

    draw.line([left_shoulder, left_hip], fill=(100, 100, 100), width=1)
    draw.line([right_shoulder, right_hip], fill=(100, 100, 100), width=1)

    print(f"\n{'t':<6}{'left candidate (x,y)':<28}{'right candidate (x,y)':<28}")
    for t in T_VALUES:
        left_candidate = interpolate(left_shoulder, left_hip, t)
        right_candidate = interpolate(right_shoulder, right_hip, t)

        print(f"{t:<6}{f'({left_candidate[0]:.1f}, {left_candidate[1]:.1f})':<28}"
              f"{f'({right_candidate[0]:.1f}, {right_candidate[1]:.1f})':<28}")

        mark(left_candidate, (255, 0, 0), 5, f"t={t}")
        mark(right_candidate, (0, 200, 255), 5, f"t={t}")

    canvas.save("underarm_derivation_test_v2.png")
    print(f"\nSaved: underarm_derivation_test_v2.png")
    print(f"\n>>> RECORD for THIS photo only:")
    print(f"    best_t: ___")
    print(f"    plausible_range: ___")
    print(f"    (Do not compare/average across photos until all are collected)")


if __name__ == "__main__":
    main()