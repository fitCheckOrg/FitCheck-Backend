"""
tryon/underarm_upload_gate.py

Turns an UnderarmLandmarkResult into an upload decision. PURE FUNCTION: no I/O, no logging, no endpoint wiring, no
database. The caller decides what to do with the decision.

Policy (deliberately plain; thresholds are NOT invented here):
  * any landmark unresolved                     -> RETAKE
  * both landmarks from the detector            -> ACCEPT
  * anything that needed the silhouette fallback,
    or a detector point that was rejected as implausible -> ACCEPT_WITH_WARNINGS
  * detector/apex disagreement is RECORDED (decision.disagreement_y_px) but never changes the action. A threshold
    needs real-user data to calibrate; two avatars are not enough.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from tryon.underarm_landmark_resolver import (
    LandmarkSource, ResolutionMethod, UnderarmLandmarkResult, SideEvidence,
)

RETAKE_CODE = "UNDERARM_NOT_DETECTED"
RETAKE_MESSAGE = (
    "We couldn't clearly see your underarms. Retake the photo standing straight, facing the camera, "
    "with your arms slightly away from your body so there is a visible gap between your arms and your torso."
)


class GateAction(str, Enum):
    ACCEPT = "accept"
    ACCEPT_WITH_WARNINGS = "accept_with_warnings"
    RETAKE = "retake"


@dataclass(frozen=True)
class GateDecision:
    action: GateAction
    revision: str
    code: Optional[str] = None                        # set only for RETAKE
    user_message: Optional[str] = None                # set only for RETAKE; safe to show the user
    warnings: Tuple[str, ...] = ()                    # for logs / support, not for the user
    reasons: Tuple[str, ...] = ()                     # why a RETAKE happened (per side), for logs / support
    method: str = ResolutionMethod.UNRESOLVED.value
    disagreement_y_px: Optional[Tuple[Optional[float], Optional[float]]] = None   # (left, right), recorded only

    @property
    def accepted(self) -> bool:
        return self.action is not GateAction.RETAKE

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "action": self.action.value, "revision": self.revision, "code": self.code,
            "user_message": self.user_message, "warnings": list(self.warnings), "reasons": list(self.reasons),
            "method": self.method,
            "disagreement_y_px": None if self.disagreement_y_px is None else list(self.disagreement_y_px),
        }
        return d


def evaluate_underarm_result(result: UnderarmLandmarkResult) -> GateDecision:
    """Never raises for a well-formed result."""
    sides = (("left", result.left, result.left_evidence), ("right", result.right, result.right_evidence))
    disagreement = (result.left_evidence.disagreement_y_px, result.right_evidence.disagreement_y_px)

    if not result.resolved:
        reasons = [f"resolver: {n}" for n in result.notes]
        for name, lm, ev in sides:
            if lm.source is LandmarkSource.UNRESOLVED:
                reasons += _why_unresolved(name, ev)
        return GateDecision(
            action=GateAction.RETAKE, revision=result.revision, code=RETAKE_CODE, user_message=RETAKE_MESSAGE,
            reasons=tuple(reasons), method=result.method.value, disagreement_y_px=disagreement,
        )

    warnings = []
    for name, lm, ev in sides:
        if lm.source is LandmarkSource.SILHOUETTE_APEX:
            warnings.append(f"{name}: used silhouette apex ({_det_status(ev)})")
    action = GateAction.ACCEPT_WITH_WARNINGS if warnings else GateAction.ACCEPT
    return GateDecision(
        action=action, revision=result.revision, warnings=tuple(warnings),
        method=result.method.value, disagreement_y_px=disagreement,
    )


def _det_status(ev: SideEvidence) -> str:
    if ev.detector.x is None:
        return "no detector point"
    return "detector point implausible: " + "; ".join(ev.detector.reasons) if ev.detector.reasons else "detector point implausible"


def _why_unresolved(name: str, ev: SideEvidence) -> list:
    out = [f"{name} detector: {r}" for r in (ev.detector.reasons or ("no detector point",))]
    out += [f"{name} apex: {r}" for r in (ev.apex.reasons or ("no apex",))]
    return out