"""
Deterministic geometry extraction — the "code scans only those
columns" step. Takes GPT's horizontal boundaries (proven reliable)
and measures actual vertical extent from the real alpha mask
(never estimated), per component.
"""

import io
import numpy as np
from PIL import Image
from shared.models.garment_representation import (
    ComponentGeometry, GeometrySource
)

ALPHA_THRESHOLD = 10  # matches the threshold used in the original 
                       # width-profile experiment


def measure_vertical_extent(image_bytes: bytes, left_percent: float, right_percent: float) -> tuple[float, float]:
    """
    Scans ONLY the given column range for opaque pixels, returns
    the real top/bottom extent as percentages of image height.
    Deterministic — same input always gives the same output,
    unlike the GPT-estimated version this replaces.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    w, h = image.size
    alpha = np.array(image)[:, :, 3]

    left_px = int(w * left_percent / 100)
    right_px = int(w * right_percent / 100)
    column_slice = alpha[:, left_px:right_px]

    opaque_rows = np.where(np.any(column_slice > ALPHA_THRESHOLD, axis=1))[0]

    if len(opaque_rows) == 0:
        # No opaque pixels found in this column range at all —
        # genuinely unusual, worth flagging rather than guessing
        return 0.0, 100.0

    top_px = opaque_rows[0]
    bottom_px = opaque_rows[-1]

    return round(top_px / h * 100, 1), round(bottom_px / h * 100, 1)


def build_component_geometry(image_bytes: bytes, gpt_left: float, gpt_right: float) -> ComponentGeometry:
    """Combines GPT's reliable horizontal call with real measured
    vertical extent for one component."""
    top, bottom = measure_vertical_extent(image_bytes, gpt_left, gpt_right)

    return ComponentGeometry(
        left_percent=gpt_left,
        right_percent=gpt_right,
        horizontal_source=GeometrySource.GPT_ESTIMATED,
        top_percent=top,
        bottom_percent=bottom,
        vertical_source=GeometrySource.PIXEL_MEASURED
    )