"""
Phase B → fitting-engine landmark interface.

This is the Phase B classification output contract — NOT a claim
that it's sufficient for all future fitting requirements. Additional
fitting inputs, if required, must be introduced explicitly rather
than reaching into Phase A internals (contours, curvature profiles,
convexity defects).

A landmark is an observation. A GarmentLandmarkSet is the collection
of observations belonging to one garment, including explicitly
established relationships between them — pairing is resolved HERE,
not left for a fitting engine to rediscover from position data.
"""

from pydantic import BaseModel, Field
from enum import Enum


class LandmarkType(str, Enum):
    HEM_CORNER = "hem_corner"
    COLLAR_SHOULDER = "collar_shoulder"
    UNDERARM = "underarm"
    AMBIGUOUS = "ambiguous"


class NormalizedPosition(BaseModel):
    x_pct: float = Field(ge=0, le=100)
    y_pct: float = Field(ge=0, le=100)


class GeometricEvidence(BaseModel):
    """The raw signals that fed into the confidence score — kept
    separate from the score so it can be inspected without
    re-deriving from contour data."""
    curvature_degrees: float
    peak_sharpness_width: int | None = None
    distance_to_left_underarm_pct: float
    distance_to_right_underarm_pct: float


class LandmarkDetectionResult(BaseModel):
    """
    One classified geometric event. Does NOT know whether it has a
    pair — that relationship lives in GarmentLandmarkSet.pairs,
    established by Phase B, not inferred downstream.
    """
    landmark_type: LandmarkType
    confidence_score: float
    position: NormalizedPosition
    geometric_evidence: GeometricEvidence
    is_ambiguous: bool

    # Diagnostic metadata — present in the contract, but consumers
    # should never depend on these for logic. Debugging/logging/UI
    # only.
    source_path_id: str
    source_contour_index: int
    classification_explanation: str


class LandmarkPair(BaseModel):
    """
    An explicitly established relationship between two landmarks
    in the same GarmentLandmarkSet, resolved by Phase B — the
    fitting engine consumes this directly, it never re-derives
    pairing from raw positions.
    """
    landmark_a_index: int  # index into GarmentLandmarkSet.landmarks
    landmark_b_index: int
    relationship: str       # e.g. "symmetric_hem_corners"
    curvature_ratio: float | None = None


class GarmentLandmarkSet(BaseModel):
    """All classified landmarks detected for one garment, plus
    established relationships between them."""
    landmarks: list[LandmarkDetectionResult]
    pairs: list[LandmarkPair]

    def by_type(self, landmark_type: LandmarkType) -> list[LandmarkDetectionResult]:
        return [
            landmark for landmark in self.landmarks
            if landmark.landmark_type == landmark_type
        ]

    def pair_for(self, landmark_index: int) -> LandmarkPair | None:
        """Returns the pair relationship involving this landmark, 
        if one exists — the fitting engine's actual entry point 
        for pairing, no position math required."""
        for pair in self.pairs:
            if landmark_index in (pair.landmark_a_index, pair.landmark_b_index):
                return pair
        return None