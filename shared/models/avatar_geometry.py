"""
AvatarGeometry — derived avatar-side geometric references.

Sits between raw MediaPipe pose_keypoints and the (not-yet-built)
fitting engine. Contains ONLY geometric points derived from
already-produced pose data — never inspects pixels, never runs
MediaPipe, never knows about garments or fitting.

Every derived reference carries its own provenance — method used,
parameter value, and status — so a future "why does 0.13 exist"
question has a real, inspectable answer rather than a buried
constant.
"""

from pydantic import BaseModel
from enum import Enum


class DerivationStatus(str, Enum):
    WORKING_HYPOTHESIS = "working_hypothesis"  # empirically supported, 
                                                  # not calibrated
    CALIBRATED = "calibrated"                   # future status, once 
                                                  # real validation exists
    UNAVAILABLE = "unavailable"                  # source landmarks 
                                                  # missing/unreliable


class DerivedPoint(BaseModel):
    """
    A single derived geometric reference, with full provenance.
    Never just a bare (x, y) — the method and status travel with 
    the point so its confidence/origin is always inspectable.
    """
    x_pct: float
    y_pct: float
    method: str
    parameter: float | None = None
    status: DerivationStatus


class AvatarGeometry(BaseModel):
    """
    Derived avatar-side references, built from raw pose_keypoints.
    Starts with only what's been investigated — left/right underarm.
    New references get added here as their own investigated
    derivations, not invented speculatively ahead of need.
    """
    left_underarm: DerivedPoint | None = None
    right_underarm: DerivedPoint | None = None