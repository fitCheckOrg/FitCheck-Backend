"""
tryon/underarm_landmark_resolver.py

Resolves the two avatar underarm landmarks from TWO independent estimates and says, per landmark, where each came
from. Pure function + plain data model: no web framework, no database, no S3, no logging, no file I/O. The caller
(AvatarRepresentation, later) supplies the image, the shoulder keypoints and whatever the SCHP detector produced.

POLICY (locked; it is a fallback, not a preference)
    per landmark:
        detector result present AND plausible   -> use the detector
        else silhouette apex found AND plausible -> use the silhouette apex
        else                                      -> UNRESOLVED
    The apex is NEVER substituted for a plausible detector result because it looks nicer. Both estimates are always
    computed and recorded in the result, so disagreement between the methods can be measured in production later
    without changing this policy.

COORDINATES AND SIDES
    Pixels of the image passed in (x right, y down) - the processed avatar image, NOT normalised pose keypoints.
    'left' / 'right' mean IMAGE-left (smaller x) and IMAGE-right (larger x), the golden-avatar convention
    (left = (61, 229), right = (162, 213)). A detector point on the wrong side of the torso centre is rejected as
    implausible, so a flipped convention shows up loudly (everything falls back to the apex) instead of silently.

THE TWO ESTIMATES
    detector         supplied by the caller as full (x, y) points, or None when it found nothing.
                     (find_semantic_underarm itself returns only a row; the x is the caller's torso outer boundary.)
    silhouette apex  the first row, below the shoulder line, where each arm separates from the torso in the avatar's
                     foreground silhouette (>= 3 consecutive rows with a background pixel between arm and torso).
                     Independent of the SCHP masks. Foreground: the alpha channel if it carries information, else
                     brightness sum > 6 (largest connected component). The threshold is deliberately LOW: at the
                     earlier 36, dark skin creases between arm and chest read as background and produced a false
                     apex ~25 rows too high on avatar #2.

PLAUSIBILITY (measured on two avatars only - these constants are starting values, not tuned truths)
    detector: finite, inside the image, correct side of the torso centre, depth below the shoulder line within
              [DEPTH_MIN, DEPTH_MAX] of the shoulder span, and within DETECTOR_SILHOUETTE_TOL_PX of the foreground.
    apex:     found, depth within [DEPTH_MIN, DEPTH_MAX], and the gap persists for at least APEX_MIN_PERSIST_FRAC of
              the shoulder span in rows (a real armpit gap continues downward; a crease artefact does not).
    No numeric 'confidence' is invented: the result reports measurements and reasons instead.

REVISION
    LANDMARK_REVISION is an explicit constant carried in every result. Change it deliberately when the resolution
    algorithm changes, so stored landmarks can be invalidated on purpose instead of served stale.

The resolver does not raise on bad data (empty image, nonsense shoulders, garbage points): it returns an UNRESOLVED
result with notes. Landmark resolution must never be the reason an avatar fails to load.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

LANDMARK_REVISION = "underarm-v2"

# --- plausibility / scan constants ------------------------------------------------------------------------------
DEPTH_MIN = 0.20                    # (underarm y - shoulder y) / shoulder span; golden 0.50, avatar #2 0.47-0.52
DEPTH_MAX = 0.80
SCAN_SPAN_FACTOR = 1.1              # scan rows from the shoulder line down to y_sh + 1.1 * span
MIN_SHOULDER_SPAN_PX = 20.0         # below this the shoulders are not pixel coordinates (or not a person)
FOREGROUND_SUM_THRESHOLD = 6        # R+G+B brightness; background of avatar images is pure black
ALPHA_THRESHOLD = 128
DETECTOR_SILHOUETTE_TOL_PX = 3
APEX_CONSECUTIVE_ROWS = 3
APEX_MIN_PERSIST_FRAC = 0.15
APEX_MIN_ARM_RUN_PX = 3             # visible arm run must be at least this wide (scaled up with the span)


# =================================================================================================================
# Return model
# =================================================================================================================
class LandmarkSource(str, Enum):
    DETECTOR = "detector"
    SILHOUETTE_APEX = "silhouette_apex"
    UNRESOLVED = "unresolved"


class ResolutionMethod(str, Enum):
    DETECTOR = "detector"                   # both landmarks from the detector
    SILHOUETTE_APEX = "silhouette_apex"     # both landmarks from the apex
    MIXED = "mixed"                         # one of each
    UNRESOLVED = "unresolved"               # at least one landmark could not be resolved


@dataclass(frozen=True)
class ResolvedLandmark:
    x: Optional[float]
    y: Optional[float]
    source: LandmarkSource

    @property
    def resolved(self) -> bool:
        return self.source is not LandmarkSource.UNRESOLVED


@dataclass(frozen=True)
class Candidate:
    """One method's estimate for one side, with the measurements behind its plausibility verdict."""
    x: Optional[float]
    y: Optional[float]
    plausible: bool
    depth: Optional[float] = None                 # (y - shoulder y) / shoulder span
    reasons: Tuple[str, ...] = ()                 # why it is NOT plausible (empty when plausible), or why it is absent
    persist_rows: Optional[int] = None            # apex only: rows with a visible arm/torso gap from the apex down


@dataclass(frozen=True)
class SideEvidence:
    """Both estimates for one side, always populated, so disagreement can be logged whatever was chosen."""
    detector: Candidate
    apex: Candidate
    disagreement_y_px: Optional[float]            # detector.y - apex.y when both have a position, else None


@dataclass(frozen=True)
class UnderarmLandmarkResult:
    left: ResolvedLandmark
    right: ResolvedLandmark
    method: ResolutionMethod
    revision: str
    left_evidence: SideEvidence
    right_evidence: SideEvidence
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    notes: Tuple[str, ...] = ()                   # resolver-level problems (bad shoulders, apex computation failed, ...)

    @property
    def resolved(self) -> bool:
        return self.left.resolved and self.right.resolved

    @property
    def points(self) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """((left x, y), (right x, y)) when both are resolved, else None."""
        if not self.resolved:
            return None
        return (self.left.x, self.left.y), (self.right.x, self.right.y)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe form for logging / storing next to the avatar."""
        return _jsonable(self)


def _jsonable(o: Any) -> Any:
    if isinstance(o, Enum):
        return o.value
    if hasattr(o, "__dataclass_fields__"):
        return {k: _jsonable(getattr(o, k)) for k in o.__dataclass_fields__}
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# =================================================================================================================
# Public entry point
# =================================================================================================================
def resolve_underarm_landmarks(
    image: Any,
    left_shoulder: Any,
    right_shoulder: Any,
    detector_left: Any = None,
    detector_right: Any = None,
) -> UnderarmLandmarkResult:
    """
    image            PIL.Image or numpy array (HxW, HxWx3 or HxWx4, uint8): the processed avatar image.
    left/right_shoulder   (x, y) pose keypoints in PIXELS of that image, either order.
    detector_left/right   the detector's (x, y) for the image-left / image-right underarm, or None if it found none.
    """
    notes = []

    # ---- shoulders -> scale and torso centre
    shoulders = _parse_shoulders(left_shoulder, right_shoulder, notes)
    if shoulders is None:
        return _unresolved_result(notes, {})
    (sx_a, sy_a), (sx_b, sy_b) = shoulders
    x_c = (sx_a + sx_b) / 2.0
    y_sh = (sy_a + sy_b) / 2.0
    span = math.hypot(sx_b - sx_a, sy_b - sy_a)

    # ---- foreground
    try:
        arr = _as_array(image)
        F, fg_source = _foreground(arr)
    except Exception as exc:                                  # noqa: BLE001 - must degrade, never raise
        notes.append(f"image could not be read: {type(exc).__name__}: {exc}")
        return _unresolved_result(notes, {"shoulder_span": span})
    H, W = F.shape
    diagnostics: Dict[str, Any] = dict(
        image_size=(int(W), int(H)), foreground_source=fg_source, foreground_area=int(F.sum()),
        shoulder_span=float(span), torso_centre_x=float(x_c), shoulder_y=float(y_sh),
        depth_range=(DEPTH_MIN, DEPTH_MAX), scan_rows=(int(max(0, y_sh)), int(min(H - 1, y_sh + SCAN_SPAN_FACTOR * span))),
    )
    if not F.any():
        notes.append("foreground is empty")

    # ---- the two estimates, always both
    apex = _apex_candidates(F, x_c, y_sh, span, notes)
    det_l = _detector_candidate(detector_left, "left", F, x_c, y_sh, span)
    det_r = _detector_candidate(detector_right, "right", F, x_c, y_sh, span)

    left_ev = SideEvidence(det_l, apex["left"], _disagreement(det_l, apex["left"]))
    right_ev = SideEvidence(det_r, apex["right"], _disagreement(det_r, apex["right"]))

    # ---- policy: detector if plausible, else apex if plausible, else unresolved
    left = _choose(left_ev)
    right = _choose(right_ev)
    return UnderarmLandmarkResult(
        left=left, right=right, method=_method(left, right), revision=LANDMARK_REVISION,
        left_evidence=left_ev, right_evidence=right_ev, diagnostics=diagnostics, notes=tuple(notes),
    )


# =================================================================================================================
# Policy
# =================================================================================================================
def _choose(ev: SideEvidence) -> ResolvedLandmark:
    if ev.detector.plausible:
        return ResolvedLandmark(ev.detector.x, ev.detector.y, LandmarkSource.DETECTOR)
    if ev.apex.plausible:
        return ResolvedLandmark(ev.apex.x, ev.apex.y, LandmarkSource.SILHOUETTE_APEX)
    return ResolvedLandmark(None, None, LandmarkSource.UNRESOLVED)


def _method(left: ResolvedLandmark, right: ResolvedLandmark) -> ResolutionMethod:
    sources = {left.source, right.source}
    if LandmarkSource.UNRESOLVED in sources:
        return ResolutionMethod.UNRESOLVED
    if sources == {LandmarkSource.DETECTOR}:
        return ResolutionMethod.DETECTOR
    if sources == {LandmarkSource.SILHOUETTE_APEX}:
        return ResolutionMethod.SILHOUETTE_APEX
    return ResolutionMethod.MIXED


def _disagreement(det: Candidate, apex: Candidate) -> Optional[float]:
    if det.y is None or apex.y is None:
        return None
    return float(det.y - apex.y)


def _unresolved_result(notes, diagnostics) -> UnderarmLandmarkResult:
    gone = Candidate(None, None, False, None, ("not evaluated: " + (notes[-1] if notes else "no reason recorded"),))
    ev = SideEvidence(gone, gone, None)
    nope = ResolvedLandmark(None, None, LandmarkSource.UNRESOLVED)
    return UnderarmLandmarkResult(nope, nope, ResolutionMethod.UNRESOLVED, LANDMARK_REVISION, ev, ev,
                                  dict(diagnostics), tuple(notes))


# =================================================================================================================
# Inputs
# =================================================================================================================
def _parse_point(p: Any) -> Optional[Tuple[float, float]]:
    try:
        a = np.asarray(p, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    if a.size != 2 or not np.all(np.isfinite(a)):
        return None
    return float(a[0]), float(a[1])


def _parse_shoulders(a: Any, b: Any, notes) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    pa, pb = _parse_point(a), _parse_point(b)
    if pa is None or pb is None:
        notes.append("shoulders are not two finite (x, y) points")
        return None
    pa, pb = sorted([pa, pb])
    span = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
    if span < MIN_SHOULDER_SPAN_PX:
        notes.append(f"shoulder span {span:.1f}px is below {MIN_SHOULDER_SPAN_PX:.0f}px: "
                     "shoulders must be pixel coordinates of the avatar image (not normalised 0-1 keypoints)")
        return None
    return pa, pb


def _as_array(image: Any) -> np.ndarray:
    if hasattr(image, "convert"):                              # PIL image (any mode)
        return np.asarray(image.convert("RGBA"))
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"unsupported image array shape {arr.shape}")
    return arr


def _foreground(arr: np.ndarray) -> Tuple[np.ndarray, str]:
    if arr.shape[2] == 4:
        a = arr[:, :, 3]
        if a.min() < ALPHA_THRESHOLD <= a.max():               # the alpha channel actually carries a silhouette
            return a >= ALPHA_THRESHOLD, "alpha"
    mask = arr[:, :, :3].astype(np.int32).sum(axis=2) > FOREGROUND_SUM_THRESHOLD
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n > 1:
        mask = lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
    return mask, f"brightness>{FOREGROUND_SUM_THRESHOLD} (largest component)"


# =================================================================================================================
# Detector candidate
# =================================================================================================================
def _detector_candidate(point: Any, side: str, F: np.ndarray, x_c: float, y_sh: float, span: float) -> Candidate:
    if point is None:
        return Candidate(None, None, False, None, ("detector returned no landmark",))
    p = _parse_point(point)
    if p is None:
        return Candidate(None, None, False, None, ("detector point is not a finite (x, y) pair",))
    x, y = p
    H, W = F.shape
    depth = (y - y_sh) / span
    reasons = []
    if not (0 <= x < W and 0 <= y < H):
        reasons.append("outside the image")
    if side == "left" and x >= x_c:
        reasons.append("'left' point is not left of the torso centre")
    if side == "right" and x <= x_c:
        reasons.append("'right' point is not right of the torso centre")
    if not (DEPTH_MIN <= depth <= DEPTH_MAX):
        reasons.append(f"depth {depth:.2f} outside {DEPTH_MIN:.2f}..{DEPTH_MAX:.2f} of the shoulder span")
    if not reasons and not _near_foreground(F, x, y, DETECTOR_SILHOUETTE_TOL_PX):
        reasons.append(f"not within {DETECTOR_SILHOUETTE_TOL_PX}px of the foreground silhouette")
    return Candidate(x, y, not reasons, float(depth), tuple(reasons))


def _near_foreground(F: np.ndarray, x: float, y: float, tol: int) -> bool:
    H, W = F.shape
    xi, yi = int(round(x)), int(round(y))
    x0, x1, y0, y1 = max(0, xi - tol), min(W, xi + tol + 1), max(0, yi - tol), min(H, yi + tol + 1)
    return bool(F[y0:y1, x0:x1].any())


# =================================================================================================================
# Silhouette apex candidate
# =================================================================================================================
def _runs_of(row: np.ndarray):
    idx = np.flatnonzero(row)
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[splits + 1]]
    ends = np.r_[idx[splits], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def _apex_candidates(F: np.ndarray, x_c: float, y_sh: float, span: float, notes) -> Dict[str, Candidate]:
    try:
        found = _find_apex(F, x_c, y_sh, span)
    except Exception as exc:                                   # noqa: BLE001 - must degrade, never raise
        notes.append(f"apex computation failed: {type(exc).__name__}: {exc}")
        failed = Candidate(None, None, False, None, ("apex computation failed",))
        return {"left": failed, "right": failed}
    out = {}
    min_persist = max(5, int(round(APEX_MIN_PERSIST_FRAC * span)))
    for side, key in (("left", "small-x"), ("right", "large-x")):
        hit = found[key]
        if hit is None:
            out[side] = Candidate(None, None, False, None, ("no arm/torso gap found in the scan window",))
            continue
        x, y, persist = hit
        depth = (y - y_sh) / span
        reasons = []
        if not (DEPTH_MIN <= depth <= DEPTH_MAX):
            reasons.append(f"depth {depth:.2f} outside {DEPTH_MIN:.2f}..{DEPTH_MAX:.2f} of the shoulder span")
        if persist < min_persist:
            reasons.append(f"gap persists for only {persist} rows (< {min_persist})")
        out[side] = Candidate(float(x), float(y), not reasons, float(depth), tuple(reasons), int(persist))
    return out


def _find_apex(F: np.ndarray, x_c: float, y_sh: float, span: float):
    H, W = F.shape
    y_start = int(max(0, y_sh))
    y_end = int(min(H - 1, y_sh + SCAN_SPAN_FACTOR * span))
    min_run = max(APEX_MIN_ARM_RUN_PX, int(round(0.025 * span)))
    sep = {"small-x": [], "large-x": []}
    for y in range(y_start, y_end + 1):
        runs = _runs_of(F[y])
        if not runs:
            continue
        inside = [r for r in runs if r[0] <= x_c <= r[1]]
        torso = inside[0] if inside else min(runs, key=lambda r: abs((r[0] + r[1]) / 2 - x_c))
        left = [r for r in runs if r[1] < torso[0] - 1 and (r[1] - r[0] + 1) >= min_run]
        right = [r for r in runs if r[0] > torso[1] + 1 and (r[1] - r[0] + 1) >= min_run]
        sep["small-x"].append((y, bool(left), ((max(r[1] for r in left) + torso[0]) / 2.0) if left else None))
        sep["large-x"].append((y, bool(right), ((torso[1] + min(r[0] for r in right)) / 2.0) if right else None))
    out = {}
    for side, lst in sep.items():
        out[side] = None
        for k in range(len(lst) - APEX_CONSECUTIVE_ROWS + 1):
            if all(lst[k + j][1] for j in range(APEX_CONSECUTIVE_ROWS)):
                persist = sum(1 for e in lst[k:] if e[1])
                out[side] = (float(lst[k][2]), float(lst[k][0]), persist)
                break
    return out