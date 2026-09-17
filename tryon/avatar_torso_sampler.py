"""
Torso-constrained contour sampler for the avatar side, replacing
the generic sample_path_between() which was built for garments'
simple closed loops and breaks on a full-body contour's complexity
(arms, legs, hand-torso gaps).

Anatomical anchors (underarm, hip) define a bounded SEARCH CORRIDOR
by side (left/right of body midline) and y-range (underarm_y to
hip_y) — the hip coordinate is used as a height/side reference, 
NOT as a literal point to nearest-match on the contour, since 
MediaPipe hip is a skeletal joint estimate, not guaranteed to lie 
on the external silhouette.
"""

import numpy as np


def sample_torso_side(contour, underarm_point, hip_point, is_left_side, num_samples):
    ua_x, ua_y = underarm_point
    hip_x, hip_y = hip_point

    y_min, y_max = min(ua_y, hip_y), max(ua_y, hip_y)
    midline_x = (ua_x + hip_x) / 2
    corridor_half_width = abs(ua_x - hip_x) + 30

    candidates = []
    for pt in contour:
        x, y = int(pt[0][0]), int(pt[0][1])
        if y_min <= y <= y_max and abs(x - midline_x) <= corridor_half_width:
            if is_left_side and x <= midline_x + corridor_half_width:
                candidates.append((x, y))
            elif not is_left_side and x >= midline_x - corridor_half_width:
                candidates.append((x, y))

    # FIRST and LAST points are the validated anchors directly —
    # not searched for. Only intermediate points come from the 
    # real contour, where torso-following silhouette is unambiguous.
    if num_samples <= 2 or len(candidates) < 2:
        return [underarm_point] + [underarm_point] * (num_samples - 2) + [hip_point]

    target_ys = np.linspace(y_min, y_max, num_samples)
    middle_targets = target_ys[1:-1]

    sampled = [underarm_point]
    for target_y in middle_targets:
        closest = min(candidates, key=lambda p: abs(p[1] - target_y))
        sampled.append(closest)
    sampled.append(hip_point)

    return sampled