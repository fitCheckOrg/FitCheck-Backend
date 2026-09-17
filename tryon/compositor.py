"""
Compositor — the final production module. Takes a FittingResult and
alpha-composites its two layers onto the avatar's original photo, in
the validated order (lower/torso first, upper/mesh second — they
share the underarm boundary, and this order is what
piecewise_continuous_test.py validated across the whole session).

Mechanical extraction only: same operations, same order, no new
interpretation of the geometry. Does no landmark detection, fitting,
resizing, or network I/O.
"""

import numpy as np
from PIL import Image


class Compositor:

    @staticmethod
    def composite(avatar_image, fitting_result):
        """
        avatar_image: PIL.Image, the original avatar photo (RGB or RGBA)
        fitting_result: FittingResult, with .upper_layer and .lower_layer
            as RGBA np.ndarrays at avatar-canvas size

        Returns: PIL.Image, same dimensions as avatar_image, with both
        garment layers alpha-composited on top in the validated order.
        """
        base = np.array(avatar_image.convert("RGBA")).astype(np.float32)

        # Lower/torso layer first — same order as every validated 
        # render this session
        for layer in [fitting_result.lower_layer, fitting_result.upper_layer]:
            alpha = layer[:, :, 3:4].astype(np.float32) / 255.0
            base = base * (1 - alpha) + layer.astype(np.float32) * alpha

        result = Image.fromarray(base.astype(np.uint8))
        return result.convert(avatar_image.mode if avatar_image.mode in ("RGB", "RGBA") else "RGB")


if __name__ == "__main__":
    # Verification: FittingEngine -> Compositor on the golden pair,
    # checked against the criteria specified
    import os
    from tryon.avatar_representation import build_avatar_representation
    from tryon.garment_representation import build_garment_representation
    from tryon.fitting_engine import FittingEngine

    GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
    AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
    OUTPUT_DIR = "outputs"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Building representations...")
    avatar = build_avatar_representation()
    garment = build_garment_representation(GOLDEN_GARMENT_ID, AVATAR_USER_ID)

    print("Fitting...")
    fitting_result = FittingEngine.fit(garment, avatar)

    print("Compositing...")
    final_image = Compositor.composite(avatar.image, fitting_result)

    # --- Verification checks, per the specified criteria ---
    print(f"\n--- Verification ---")
    print(f"Avatar dimensions:      {avatar.image.size}")
    print(f"Final image dimensions: {final_image.size}")
    print(f"Dimensions match: {avatar.image.size == final_image.size}")

    print(f"Avatar mode: {avatar.image.mode}")
    print(f"Final mode:  {final_image.mode}")

    final_arr = np.array(final_image.convert("RGB"))
    avatar_arr = np.array(avatar.image.convert("RGB"))
    diff = np.abs(final_arr.astype(int) - avatar_arr.astype(int)).sum(axis=2)
    changed_pixels = (diff > 5).sum()
    print(f"\nPixels changed from original avatar (garment region): {changed_pixels}")

    # "Avatar outside unchanged": check a corner region far from the 
    # garment (e.g. top-left 20x20, which should be background/head area)
    corner_diff = diff[:20, :20]
    print(f"Top-left 20x20 corner max diff (should be ~0, outside garment area): {corner_diff.max()}")

    final_image.save(f"{OUTPUT_DIR}/compositor_verification.png")
    print(f"\nSaved: {OUTPUT_DIR}/compositor_verification.png")
    print(f"\n>>> Compare this directly against the earlier validated "
          f"piecewise_continuous_test.py golden render — should be visually "
          f"identical (same math, mechanically extracted, not reinterpreted).")