"""
Arm-aware torso sampler. Uses MediaPipe's elbow/wrist keypoints
(already available, previously unused) to build an exclusion zone
around the arm's actual line, rejecting contour candidates that
are closer to the arm than to the expected torso side.
"""

import numpy as np


def distance_to_line_segment(point, line_start, line_end):
    """Perpendicular distance from a point to a line segment."""
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end

    dx, dy = x2 - x1, y2 - y1
    length_sq = dx*dx + dy*dy
    if length_sq == 0:
        return np.hypot(px - x1, py - y1)

    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
    proj_x, proj_y = x1 + t * dx, y1 + t * dy
    return np.hypot(px - proj_x, py - proj_y)


def sample_torso_side_arm_aware(contour, underarm_point, hip_point,
                                   shoulder_point, elbow_point, wrist_point,
                                   is_left_side, num_samples,
                                   arm_exclusion_margin=15):
    """
    Same corridor logic as before, but now rejects any candidate
    closer to the arm's shoulder->elbow->wrist path than
    arm_exclusion_margin pixels — using real skeletal data instead
    of guessing from silhouette shape alone.
    """
    ua_x, ua_y = underarm_point
    hip_x, hip_y = hip_point

    y_min, y_max = min(ua_y, hip_y), max(ua_y, hip_y)
    midline_x = (ua_x + hip_x) / 2
    corridor_half_width = abs(ua_x - hip_x) + 30

    arm_segments = [(shoulder_point, elbow_point), (elbow_point, wrist_point)]

    candidates = []
    for pt in contour:
        x, y = int(pt[0][0]), int(pt[0][1])
        if not (y_min <= y <= y_max and abs(x - midline_x) <= corridor_half_width):
            continue
        if is_left_side and x > midline_x + corridor_half_width:
            continue
        if not is_left_side and x < midline_x - corridor_half_width:
            continue

        # NEW: reject if too close to the arm's real line
        min_dist_to_arm = min(
            distance_to_line_segment((x, y), seg[0], seg[1]) for seg in arm_segments
        )
        if min_dist_to_arm < arm_exclusion_margin:
            continue

        candidates.append((x, y))

    if len(candidates) < 2:
        # Fall back to anchors only if arm exclusion removed everything
        return [underarm_point] + [underarm_point] * (num_samples - 2) + [hip_point]

    target_ys = np.linspace(y_min, y_max, num_samples)
    middle_targets = target_ys[1:-1]

    sampled = [underarm_point]
    for target_y in middle_targets:
        closest = min(candidates, key=lambda p: abs(p[1] - target_y))
        sampled.append(closest)
    sampled.append(hip_point)

    return sampled