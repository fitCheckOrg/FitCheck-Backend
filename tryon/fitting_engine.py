"""
FittingEngine — pure correspondence, mesh construction, and
deformation. Owns none of: downloading, SCHP, landmark detection,
avatar analysis, or final compositing onto the avatar photo.

Consumes already-built GarmentRepresentation + AvatarRepresentation.
Does not rediscover anything they already know.

V1 scope, explicit: only the shoulders_available=True path is
implemented. shoulders_available=False raises UnsupportedGarmentError
— this is a deliberate V1 rejection, not a silent degraded fallback.
An underarm-only path is real future work, not yet designed or
validated, and will not be added here until it earns that status
the same way the mesh path did.
"""

import numpy as np
import cv2
from PIL import Image

from tryon.piecewise_continuous_test import (
    warp_triangle, composite_triangle, solve_similarity_2point
)


class UnsupportedGarmentError(Exception):
    """Raised when the garment's representation doesn't support the
    only fitting path currently implemented. Not a bug — an honest
    signal that V1 doesn't yet cover this case."""
    pass


class FittingResult:
    """
    Holds the deformed garment as separate, uncomposited RGBA layers
    at avatar-canvas size. Compositing these onto the avatar's actual
    photo — and deciding the blend order — is the Compositor's job,
    not this class's.
    """
    def __init__(self, upper_layer, lower_layer):
        self.upper_layer = upper_layer  # np.ndarray, RGBA, mesh-warped shoulder/underarm region
        self.lower_layer = lower_layer  # np.ndarray, RGBA, underarm-anchored torso region


class FittingEngine:

    @staticmethod
    def fit(garment, avatar):
        """
        garment: GarmentRepresentation
        avatar: AvatarRepresentation

        Returns: FittingResult

        Raises: UnsupportedGarmentError if garment.shoulders_available
        is False — V1 has no validated path for this case yet.
        """
        if not garment.shoulders_available:
            raise UnsupportedGarmentError(
                "This garment's shoulder landmarks are unavailable "
                "(shoulders_available=False). V1 FittingEngine has no "
                "validated fitting path without shoulder correspondence "
                "— an underarm-only strategy is future work, not yet "
                "designed or validated. Not falling back silently."
            )

        aw, ah = avatar.image.size

        # Pair by physical x-position (garment labels are image-space,
        # avatar labels are anatomical-space — established earlier
        # this project; pairing must be by coordinate, not label name)
        g_small_ua = garment.left_underarm if garment.left_underarm[0] < garment.right_underarm[0] else garment.right_underarm
        g_large_ua = garment.right_underarm if garment.left_underarm[0] < garment.right_underarm[0] else garment.left_underarm
        g_small_sh = garment.left_shoulder if garment.left_shoulder[0] < garment.right_shoulder[0] else garment.right_shoulder
        g_large_sh = garment.right_shoulder if garment.left_shoulder[0] < garment.right_shoulder[0] else garment.left_shoulder

        a_small_ua = avatar.left_underarm if avatar.left_underarm[0] < avatar.right_underarm[0] else avatar.right_underarm
        a_large_ua = avatar.right_underarm if avatar.left_underarm[0] < avatar.right_underarm[0] else avatar.left_underarm
        a_small_sh = avatar.left_shoulder if avatar.left_shoulder[0] < avatar.right_shoulder[0] else avatar.right_shoulder
        a_large_sh = avatar.right_shoulder if avatar.left_shoulder[0] < avatar.right_shoulder[0] else avatar.left_shoulder

        # --- Continuous shared-boundary mesh (validated path, 6/7 
        # clean on visual review) — math unchanged from research code ---
        src_tris = [[g_small_sh, g_large_sh, g_small_ua], [g_large_sh, g_small_ua, g_large_ua]]
        dst_tris = [[a_small_sh, a_large_sh, a_small_ua], [a_large_sh, a_small_ua, a_large_ua]]

        garment_arr = np.array(garment.image)
        upper_layer = np.zeros((ah, aw, 4), dtype=np.uint8)
        for src_tri, dst_tri in zip(src_tris, dst_tris):
            warped, rect = warp_triangle(garment_arr, src_tri, dst_tri, (aw, ah))
            composite_triangle(upper_layer, warped, rect)

        # --- Underarm-anchored lower/torso transform (locked baseline,
        # math unchanged) ---
        torso_matrix = solve_similarity_2point([g_small_ua, g_large_ua], [a_small_ua, a_large_ua])

        x1, y1 = g_small_ua
        x2, y2 = g_large_ua
        yy, xx = np.indices(garment.mask.shape)
        boundary_y = y1 + (xx - x1) * (y2 - y1) / (x2 - x1)
        lower_keep = yy >= boundary_y
        torso_zone = garment_arr.copy()
        torso_zone[:, :, 3] = np.where(lower_keep, torso_zone[:, :, 3], 0)

        dst_buffer = np.zeros((ah, aw, 4), dtype=np.uint8)  # pre-allocated — avoids the uninitialized-memory warpAffine bug found earlier
        lower_layer = cv2.warpAffine(
            torso_zone, torso_matrix, (aw, ah), dst=dst_buffer,
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT
        )

        return FittingResult(upper_layer=upper_layer, lower_layer=lower_layer)


if __name__ == "__main__":
    # Sanity check against the golden pair
    from tryon.avatar_representation import build_avatar_representation
    from tryon.garment_representation import build_garment_representation

    GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
    AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"

    print("Building representations...")
    avatar = build_avatar_representation()
    garment = build_garment_representation(GOLDEN_GARMENT_ID, AVATAR_USER_ID)

    print(f"Avatar underarms: L={avatar.left_underarm}  R={avatar.right_underarm}")
    print(f"Garment shoulders_available: {garment.shoulders_available}")

    print("\nFitting...")
    result = FittingEngine.fit(garment, avatar)
    print(f"Upper layer shape: {result.upper_layer.shape}")
    print(f"Lower layer shape: {result.lower_layer.shape}")
    print(f"Upper layer nonzero pixels: {(result.upper_layer[:,:,3] > 0).sum()}")
    print(f"Lower layer nonzero pixels: {(result.lower_layer[:,:,3] > 0).sum()}")
    print("\n>>> FittingEngine ran successfully — returns uncomposited "
          "layers only, no image I/O, no landmark detection, no "
          "compositing onto the avatar photo.")