"""
First production-path test. Calls run_tryon_pipeline() directly —
no API, no HTTP. Verifies the full chain: Supabase-backed
representations -> FittingEngine -> Compositor -> real S3 upload.

Run: python -m tryon.test_golden_production_pipeline
"""

import asyncio
import requests
from PIL import Image
import io

from tryon.tryon_pipeline import run_tryon_pipeline
from shared.exceptions import FitCheckException

GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"


async def main():
    print("Running production tryon_pipeline against golden avatar + garment...")
    print("(No API involved — calling run_tryon_pipeline() directly)\n")

    try:
        result = await run_tryon_pipeline(
            garment_id=GOLDEN_GARMENT_ID,
            user_id=AVATAR_USER_ID,
        )
    except FitCheckException as e:
        print(f"FAILED — FitCheckException raised: code={e.code} status={e.status_code} message={e.message}")
        if e.code == "GARMENT_UNSUPPORTED":
            print(">>> This means shoulders_available was False for the golden garment "
                  ">>> — that would itself be a real regression, since it's always "
                  ">>> been True for this exact garment all session.")
        return
    except Exception as e:
        print(f"FAILED — unexpected exception NOT wrapped as FitCheckException: {type(e).__name__}: {e}")
        print(">>> This means something in the chain raised a raw exception that "
              ">>> escaped the pipeline's try/except — worth checking why.")
        return

    print(f"TryOnResult returned:")
    print(f"  url: {result.url}")
    print(f"  key: {result.key}")

    checks = []
    checks.append(("TryOnResult.url exists and is non-empty", bool(result.url)))
    checks.append(("TryOnResult.key exists and is non-empty", bool(result.key)))

    print(f"\nDownloading uploaded image from S3 to verify...")
    try:
        resp = requests.get(result.url, timeout=10)
        checks.append(("S3 URL is downloadable (200 OK)", resp.status_code == 200))
        img = Image.open(io.BytesIO(resp.content))
        checks.append(("Image dimensions are (223, 673)", img.size == (223, 673)))
        checks.append(("Image mode is RGB", img.mode == "RGB"))
    except Exception as e:
        checks.append((f"Download/verify failed: {e}", False))

    print(f"\n{'='*60}\nMECHANICAL CHECKS\n{'='*60}")
    all_passed = True
    for description, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {description}")
        if not passed:
            all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print("ALL MECHANICAL CHECKS PASSED.")
        print(f"\n>>> Manually download and view {result.url} to visually confirm "
              f"it matches the already-verified golden render (same shoulders "
              f"level, same collar centered, no distortion, no floating).")
    else:
        print("SOME CHECKS FAILED — see above. Do not proceed to API wiring "
              "until these pass.")


if __name__ == "__main__":
    asyncio.run(main())