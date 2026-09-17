"""
Extracts the garment's coordinate system from already-validated
Layer 0-3 geometry: underarms, hem. Collar deliberately omitted for
this first prototype — not yet needed for a torso-only affine fit.
"""

import numpy as np


def get_garment_coordinate_system(left_underarm, right_underarm, hem_point):
    """
    Returns the garment's own reference frame: underarm midpoint
    (anchor), underarm-to-underarm width (scale reference), and
    underarm-to-hem height (torso length reference).
    """
    mid_x = (left_underarm[0] + right_underarm[0]) / 2
    mid_y = (left_underarm[1] + right_underarm[1]) / 2
    width = np.hypot(right_underarm[0] - left_underarm[0],
                      right_underarm[1] - left_underarm[1])
    height = hem_point[1] - mid_y

    return {
        "anchor": (mid_x, mid_y),
        "width": width,
        "height": height,
        "left_underarm": left_underarm,
        "right_underarm": right_underarm,
        "hem": hem_point,
    }