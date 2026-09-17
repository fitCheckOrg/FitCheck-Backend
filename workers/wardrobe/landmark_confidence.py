"""
Landmark confidence scoring — Phase A design, not yet wired into
any pipeline. Scores individual geometric events (curvature peaks)
on plausibility as real garment landmarks, using corroborating
signals rather than a single pass/fail gate.

Core principle from tonight's investigation: pairing is
corroboration, not validation. A strong isolated corner (confirmed
real via direct raw-contour inspection) must not be penalized for
lacking a symmetric counterpart — real garment photos aren't always
physically symmetric.
"""

import numpy as np


def compute_peak_sharpness(curvature_profile: np.ndarray, peak_idx: int, threshold_ratio: float = 0.5) -> int:
    """
    Measures how narrow the elevated-curvature region is around a
    peak — a genuinely different signal from the peak's raw
    magnitude. Counts contour points on each side that stay above
    threshold_ratio * peak_value before dropping below it.

    A true sharp corner: small width (curvature spikes and returns
    to baseline quickly).
    A gradual curve (e.g. rounding a shoulder): large width
    (curvature stays elevated over many points).

    Returns total width in contour-index units — SMALLER is sharper.
    """
    peak_value = curvature_profile[peak_idx]
    threshold = peak_value * threshold_ratio
    n = len(curvature_profile)

    left_width = 0
    i = peak_idx
    while i > 0 and curvature_profile[i] >= threshold:
        left_width += 1
        i -= 1

    right_width = 0
    i = peak_idx
    while i < n - 1 and curvature_profile[i] >= threshold:
        right_width += 1
        i += 1

    return left_width + right_width


def find_symmetric_pair(event: dict, all_events: list[dict],
                          y_tolerance_pct: float = 8.0,
                          x_symmetry_tolerance_pct: float = 10.0,
                          curvature_ratio_threshold: float = 2.0) -> dict | None:
    """
    Reuses the exact pairing logic validated tonight (position
    symmetry + curvature-ratio filter). Returns the best matching
    pair if one exists, or None — absence is NOT a failure signal,
    just means no corroboration is available for this event.
    """
    best_pair = None
    best_score = float('inf')  # lower combined diff = better match

    for other in all_events:
        if other is event:
            continue

        y_diff = abs(event["y_pct"] - other["y_pct"])
        if y_diff > y_tolerance_pct:
            continue

        mirror_x = 100 - event["x_pct"]
        x_symmetry_diff = abs(mirror_x - other["x_pct"])
        if x_symmetry_diff > x_symmetry_tolerance_pct:
            continue

        curv_ratio = max(event["curvature"], other["curvature"]) / max(
            min(event["curvature"], other["curvature"]), 0.1
        )
        if curv_ratio >= curvature_ratio_threshold:
            continue

        combined_diff = y_diff + x_symmetry_diff
        if combined_diff < best_score:
            best_score = combined_diff
            best_pair = {**other, "curvature_ratio": round(curv_ratio, 2)}

    return best_pair


def landmark_confidence(event: dict, all_events: list[dict], curvature_profile: np.ndarray) -> tuple[float, list[str]]:
    """
    Scores a single geometric event's plausibility as a real
    landmark. Every feature here was individually examined during
    tonight's investigation — this combines them rather than
    gating on any single one.

    Returns (score, reasons) — reasons are for human review, not
    just a black-box number.
    """
    score = 0.0
    reasons = []

    # 1. Raw curvature magnitude — validated as a real signal 
    #    (peak diagnostic), not sufficient alone
    if event["curvature"] > 80:
        score += 30
        reasons.append(f"strong curvature ({event['curvature']}°)")
    elif event["curvature"] > 40:
        score += 15
        reasons.append(f"moderate curvature ({event['curvature']}°)")

    # 2. Position near a vertical extreme — consistent with 
    #    hem/collar zones across every garment tested tonight
    distance_from_nearest_extreme = min(event["y_pct"], 100 - event["y_pct"])
    extreme_bonus = max(0, 20 * (1 - distance_from_nearest_extreme / 20))
    if extreme_bonus > 0:
        score += extreme_bonus
        reasons.append(f"near vertical extreme (bonus={extreme_bonus:.1f}, y={event['y_pct']}%)")
    # 3. Distance from underarm — validated primitive (9/9 
    #    qualified). Far from underarm = less likely to be 
    #    underarm-rediscovery noise
    nearest_underarm_dist = min(
        event.get("dist_to_left_underarm_pct", 100),
        event.get("dist_to_right_underarm_pct", 100)
    )
    if nearest_underarm_dist > 40:
        score += 10
        reasons.append("far from underarm — distinct from underarm signal")
    elif nearest_underarm_dist < 5:
        score -= 20
        reasons.append("PENALIZED: very close to underarm — likely re-detecting known landmark, not new")
    # 4. Bilateral pairing — CORROBORATION ONLY. Adds confidence 
    #    when present. Absence adds nothing, but never subtracts —
    #    this is the core correction from tonight's raw-geometry finding.
    pair = find_symmetric_pair(event, all_events)
    if pair:
        score += 25
        reasons.append(f"symmetric pair found (ratio={pair['curvature_ratio']})")
    else:
        reasons.append("no symmetric pair — not penalized (real garments can be locally asymmetric)")

    # 5. Peak sharpness — genuinely distinct from magnitude. A 
    #    narrow, sharp spike is more consistent with a true corner 
    #    (side seam meeting hem) than a broad, gradual curve.
    if "peak_idx" in event and curvature_profile is not None:
        width = compute_peak_sharpness(curvature_profile, event["peak_idx"])
        if width < 10:
            score += 15
            reasons.append(f"sharp, narrow peak (width={width})")
        elif width > 30:
            reasons.append(f"broad/gradual curve (width={width}) — may be a rounded transition, not a corner")

    return round(score, 1), reasons