"""
Checks whether the original (pre-crop) photo's aspect ratio
differs from the processed (post-crop) photo's aspect ratio for
this avatar. If pose_keypoints are normalized against the
ORIGINAL image but reapplied to the PROCESSED image's dimensions,
a ratio mismatch would directly explain why landmarks land in the
wrong relative position — a structural bug, not a formula error.

Run: python -m tryon.aspect_ratio_mismatch_check
"""

import io
import requests
from PIL import Image

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"


def main():
    avatar_resp = requests.get(
        f"http://127.0.0.1:8000/api/avatar/me?user_id={AVATAR_USER_ID}"
    ).json()["data"]

    original_url = avatar_resp.get("original_photo_url")
    processed_url = avatar_resp.get("processed_photo_url")

    print(f"original_photo_url: {original_url}")
    print(f"processed_photo_url: {processed_url}")

    original_bytes = requests.get(original_url).content
    processed_bytes = requests.get(processed_url).content

    original_img = Image.open(io.BytesIO(original_bytes))
    processed_img = Image.open(io.BytesIO(processed_bytes))

    ow, oh = original_img.size
    pw, ph = processed_img.size

    original_ratio = ow / oh
    processed_ratio = pw / ph

    print(f"\nORIGINAL:  {ow} x {oh}  (aspect ratio {original_ratio:.4f})")
    print(f"PROCESSED: {pw} x {ph}  (aspect ratio {processed_ratio:.4f})")

    ratio_diff = abs(original_ratio - processed_ratio)
    ratio_diff_pct = ratio_diff / original_ratio * 100

    print(f"\nAspect ratio difference: {ratio_diff:.4f} ({ratio_diff_pct:.1f}%)")

    if ratio_diff_pct > 5:
        print(f"\n⚠ SIGNIFICANT MISMATCH — pose_keypoints normalized against "
              f"the original {ow}x{oh} image would land in visibly wrong "
              f"relative positions when reapplied to the {pw}x{ph} processed "
              f"image. This directly explains the landmark misplacement.")
    else:
        print(f"\nRatios are close — aspect ratio mismatch is likely NOT "
              f"the primary cause. The problem may be elsewhere "
              f"(pose_keypoints accuracy itself, or another bug).")


if __name__ == "__main__":
    main()