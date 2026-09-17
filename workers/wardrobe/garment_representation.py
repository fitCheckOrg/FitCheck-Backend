from shared.logger import get_logger
from shared.models.garment_representation import (
    GarmentRepresentation,
    GarmentRegion,
    GarmentCollar,
    MAX_AUTO_NORMALIZE_PERCENT
)

logger = get_logger(__name__)


def build_garment_representation(raw: dict) -> GarmentRepresentation:
    """
    Parses and validates a raw GPT-4o Vision response into a
    GarmentRepresentation. Never raises on bad AI output — always
    returns a representation, with is_valid=False and explanatory
    notes if something's wrong, rather than crashing the caller.
    """
    notes = []

    try:
        garment_type = raw.get("garment_type", "unknown")
        observations = raw.get("observations", "")
        hem_percent = raw.get("hem_percent")

        if hem_percent is None:
            notes.append("Missing hem_percent")
            return _invalid_representation(garment_type, observations, notes)

        collar = None
        if raw.get("collar"):
            try:
                collar = GarmentCollar(**raw["collar"])
            except Exception as e:
                notes.append(f"Invalid collar data, dropped: {e}")

        torso_raw = raw.get("torso_region")
        if not torso_raw:
            notes.append("Missing torso_region — cannot build representation")
            return _invalid_representation(garment_type, observations, notes)

        try:
            torso = GarmentRegion(**torso_raw)
        except Exception as e:
            notes.append(f"Invalid torso_region: {e}")
            return _invalid_representation(garment_type, observations, notes)

        left_sleeve = _parse_optional_region(raw.get("left_sleeve_region"), "left_sleeve_region", notes)
        right_sleeve = _parse_optional_region(raw.get("right_sleeve_region"), "right_sleeve_region", notes)

        was_normalized = False

        if left_sleeve is not None:
            left_sleeve, torso, normalized, note = _reconcile_boundary(
                left_sleeve, torso, "left_sleeve_region", "torso_region"
            )
            if normalized:
                was_normalized = True
            if note:
                notes.append(note)

        if right_sleeve is not None:
            torso, right_sleeve, normalized, note = _reconcile_boundary(
                torso, right_sleeve, "torso_region", "right_sleeve_region"
            )
            if normalized:
                was_normalized = True
            if note:
                notes.append(note)

        # Check outer-edge coverage — do the leftmost/rightmost
        # regions reach the image edges (0/100)? Mirrors the
        # adjacent-region logic: small gaps get auto-normalized
        # (extend the outer region to the edge), larger ones flagged.
        if left_sleeve is not None:
            edge_gap = left_sleeve.left_percent - 0
            if edge_gap > 0:
                if edge_gap <= MAX_AUTO_NORMALIZE_PERCENT:
                    left_sleeve = GarmentRegion(left_percent=0, right_percent=left_sleeve.right_percent)
                    was_normalized = True
                    notes.append(f"Normalized {edge_gap:.1f}pp left-edge gap — extended left_sleeve_region to 0%")
                else:
                    notes.append(f"INVALID: {edge_gap:.1f}pp unclaimed at left edge, exceeds threshold")

        if right_sleeve is not None:
            edge_gap = 100 - right_sleeve.right_percent
            if edge_gap > 0:
                if edge_gap <= MAX_AUTO_NORMALIZE_PERCENT:
                    right_sleeve = GarmentRegion(left_percent=right_sleeve.left_percent, right_percent=100)
                    was_normalized = True
                    notes.append(f"Normalized {edge_gap:.1f}pp right-edge gap — extended right_sleeve_region to 100%")
                else:
                    notes.append(f"INVALID: {edge_gap:.1f}pp unclaimed at right edge, exceeds threshold")

        is_valid = not any("INVALID:" in n for n in notes)

        return GarmentRepresentation(
            garment_type=garment_type,
            observations=observations,
            collar=collar,
            torso_region=torso,
            left_sleeve_region=left_sleeve,
            right_sleeve_region=right_sleeve,
            hem_percent=hem_percent,
            is_valid=is_valid,
            validation_notes=notes,
            was_normalized=was_normalized
        )

    except Exception as e:
        logger.exception("Unexpected error building garment representation")
        return _invalid_representation(
            raw.get("garment_type", "unknown"),
            raw.get("observations", ""),
            [f"INVALID: unexpected parse error: {e}"]
        )


def _parse_optional_region(raw, field_name, notes):
    if not raw:
        return None
    try:
        return GarmentRegion(**raw)
    except Exception as e:
        notes.append(f"Invalid {field_name}, dropped: {e}")
        return None


def _reconcile_boundary(left_region, right_region, left_name, right_name):
    gap = right_region.left_percent - left_region.right_percent

    if gap == 0:
        return left_region, right_region, False, None

    abs_gap = abs(gap)
    kind = "gap" if gap > 0 else "overlap"

    if abs_gap <= MAX_AUTO_NORMALIZE_PERCENT:
        midpoint = (left_region.right_percent + right_region.left_percent) / 2
        normalized_left = GarmentRegion(left_percent=left_region.left_percent, right_percent=midpoint)
        normalized_right = GarmentRegion(left_percent=midpoint, right_percent=right_region.right_percent)
        note = f"Normalized {abs_gap:.1f}pp {kind} between {left_name} and {right_name} — snapped to midpoint {midpoint:.1f}%"
        return normalized_left, normalized_right, True, note

    note = f"INVALID: {abs_gap:.1f}pp {kind} between {left_name} and {right_name} exceeds auto-normalize threshold ({MAX_AUTO_NORMALIZE_PERCENT}pp) — not auto-corrected"
    return left_region, right_region, False, note


def _invalid_representation(garment_type, observations, notes):
    return GarmentRepresentation(
        garment_type=garment_type,
        observations=observations,
        torso_region=GarmentRegion(left_percent=0, right_percent=100),
        hem_percent=100,
        is_valid=False,
        validation_notes=[f"INVALID: {n}" if "INVALID:" not in n else n for n in notes]
    )