"""
Captures reference fixtures for visual regression. Run this ONLY
after visually verifying each render is correct — never as an
automated side effect. This is how new fixtures get created or
existing ones get deliberately updated.

Run: python -m tests.regression.capture_fixtures
"""

import os
from tryon.avatar_representation import build_avatar_representation
from tryon.garment_representation import build_garment_representation
from tryon.fitting_engine import FittingEngine
from tryon.compositor import Compositor
from tests.regression.test_golden_pipeline import GARMENTS, AVATAR_USER_ID, FIXTURE_DIR


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    avatar = build_avatar_representation()

    for g in GARMENTS:
        if not g["shoulders_available"]:
            print(f"Skipping {g['label']} (no shoulders, no render to capture)")
            continue

        print(f"Rendering {g['label']}...")
        garment = build_garment_representation(g["id"], AVATAR_USER_ID)
        result = FittingEngine.fit(garment, avatar)
        final = Compositor.composite(avatar.image, result)

        path = os.path.join(FIXTURE_DIR, f"{g['label']}_reference.png")
        final.save(path)
        print(f"  Saved: {path}  (status: {g['known_status']})")

    print(f"\n>>> IMPORTANT: visually verify each saved fixture before trusting "
          f"it as a regression baseline. Do not skip this — this is exactly "
          f"the discipline that caught the fresh_3/fresh_7 stale-render issue.")


if __name__ == "__main__":
    main()