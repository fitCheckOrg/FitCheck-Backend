"""
AvatarGeometryDerivation.

Derives AvatarGeometry from raw MediaPipe pose_keypoints. This
module explicitly does NOT:
  - know about garments
  - know about fitting
  - know about compositing
  - inspect image pixels
  - run MediaPipe

It only derives geometry from already-produced avatar landmarks —
the one, narrow responsibility this layer exists for.
"""

from shared.models.avatar_geometry import AvatarGeometry, DerivedPoint, DerivationStatus

# Empirically supported working hypothesis, NOT a calibrated 
# anatomical constant. Derived from 3 real avatar photos, arm-
# separation and resolution varying across all three:
#   Photo 1 (relaxed arms, occluded):     plausible range 0.10-0.15
#   Photo 2 (shirtless, slight gap, low-res): plausible range 0.12-0.16
#   Photo 3 (clear gap, good resolution): plausible range 0.08-0.14
# Overlap across all three: ~0.12-0.14. See underarm_derivation_test.py
# for the actual test script and raw evidence.
UNDERARM_INTERPOLATION_T = 0.13
UNDERARM_METHOD_NAME = "shoulder_hip_interpolation"


def _interpolate(shoulder: dict, hip: dict, t: float) -> tuple[float, float]:
    """U = S + t(H - S), on normalized 0-1 coordinates (matches 
    MediaPipe's own output format directly — no pixel conversion 
    needed here, that happens downstream if/when needed)."""
    x = shoulder["x"] + t * (hip["x"] - shoulder["x"])
    y = shoulder["y"] + t * (hip["y"] - shoulder["y"])
    return x, y


def derive_avatar_geometry(pose_keypoints: dict) -> AvatarGeometry:
    """
    Builds AvatarGeometry from raw pose_keypoints. Returns
    left_underarm/right_underarm as None (not a crash) if the
    required source landmarks are missing — matches the same
    non-blocking philosophy used throughout Phase A/B rather than
    failing the whole derivation over one missing reference.
    """
    left_underarm = None
    right_underarm = None

    if "left_shoulder" in pose_keypoints and "left_hip" in pose_keypoints:
        x, y = _interpolate(
            pose_keypoints["left_shoulder"], pose_keypoints["left_hip"],
            UNDERARM_INTERPOLATION_T
        )
        left_underarm = DerivedPoint(
            x_pct=round(x * 100, 2), y_pct=round(y * 100, 2),
            method=UNDERARM_METHOD_NAME,
            parameter=UNDERARM_INTERPOLATION_T,
            status=DerivationStatus.WORKING_HYPOTHESIS
        )

    if "right_shoulder" in pose_keypoints and "right_hip" in pose_keypoints:
        x, y = _interpolate(
            pose_keypoints["right_shoulder"], pose_keypoints["right_hip"],
            UNDERARM_INTERPOLATION_T
        )
        right_underarm = DerivedPoint(
            x_pct=round(x * 100, 2), y_pct=round(y * 100, 2),
            method=UNDERARM_METHOD_NAME,
            parameter=UNDERARM_INTERPOLATION_T,
            status=DerivationStatus.WORKING_HYPOTHESIS
        )

    return AvatarGeometry(left_underarm=left_underarm, right_underarm=right_underarm)