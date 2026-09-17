"""
Phase B — deterministic landmark classification.

Takes a candidate event that has ALREADY passed Phase A confidence
scoring and attempts a coarse classification: which structural
category does it most likely belong to. Rule-based only, using
signals already validated in Phase A — no new geometric features,
no ML.

Explicitly designed to say "ambiguous" rather than force a
guess — an honest unknown is more useful downstream than a
confident-looking wrong answer.
"""

from enum import Enum


class LandmarkCategory(str, Enum):
    HEM_CORNER = "hem_corner"
    COLLAR_SHOULDER = "collar_shoulder"
    UNDERARM = "underarm"
    AMBIGUOUS = "ambiguous"


def classify_landmark(event: dict, confidence_score: float, reasons: list[str]) -> tuple[LandmarkCategory, str]:
    """
    Deterministic classification based on position zone + Phase A
    signals already computed. Returns (category, explanation) —
    explanation is for human review, same principle as Phase A's
    reasons list.
    """
    y = event["y_pct"]

    # Already flagged by Phase A as underarm-adjacent — don't 
    # reclassify, just label it what Phase A already determined
    if any("PENALIZED" in r and "underarm" in r for r in reasons):
        return LandmarkCategory.UNDERARM, "Phase A flagged as underarm-adjacent"

    # Confidence floor — below this, don't attempt classification 
    # at all, regardless of position. An unreliable event 
    # classified confidently is worse than an unclassified one.
    if confidence_score < 50:
        return LandmarkCategory.AMBIGUOUS, f"confidence too low ({confidence_score}) to classify"

    if y > 80:
        return LandmarkCategory.HEM_CORNER, f"y={y}% in validated hem-zone range"

    if y < 15:
        return LandmarkCategory.COLLAR_SHOULDER, f"y={y}% in validated collar/shoulder-zone range"

    # Mid-height, confident, but no zone match — this is exactly 
    # the honest "known limitation" bucket from Phase A, not a 
    # new failure. B#1 on the rugby shirt (y=10.2%... wait, that's 
    # actually collar zone) — the real ambiguous case is something 
    # like a mid-torso event with no zone match at all.
    return LandmarkCategory.AMBIGUOUS, f"y={y}% — confident event but no zone match, needs further signal"