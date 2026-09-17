"""
Generic contour-sampling utility. Given a contour and two target
points, finds the nearest contour vertex to each, determines which
traversal direction between them is shorter (reusing the validated
"smaller arc wins" logic from the cuff bilateral-resolution work),
and samples K evenly-spaced intermediate points along that real
path — not a fabricated straight line.
"""

import numpy as np


def find_nearest_contour_index(contour, target_point):
    tx, ty = target_point
    best_idx, best_dist = None, float('inf')
    for i, pt in enumerate(contour):
        x, y = pt[0]
        dist = np.hypot(x - tx, y - ty)
        if dist < best_dist:
            best_dist, best_idx = dist, i
    return best_idx


def contour_arc_distances(contour, start_idx):
    n = len(contour)
    forward, backward = {}, {}
    dist = 0.0
    for step in range(n):
        idx = (start_idx + step) % n
        forward[idx] = dist
        next_idx = (idx + 1) % n
        dist += np.linalg.norm(contour[next_idx][0].astype(float) - contour[idx][0].astype(float))
    dist = 0.0
    for step in range(n):
        idx = (start_idx - step) % n
        backward[idx] = dist
        prev_idx = (idx - 1) % n
        dist += np.linalg.norm(contour[prev_idx][0].astype(float) - contour[idx][0].astype(float))
    return forward, backward


def sample_path_between(contour, start_point, end_point, num_samples):
    """
    Finds the real contour path between start_point and end_point
    (whichever direction is shorter), then samples num_samples
    evenly-spaced points along that ACTUAL path.
    """
    start_idx = find_nearest_contour_index(contour, start_point)
    end_idx = find_nearest_contour_index(contour, end_point)

    forward, backward = contour_arc_distances(contour, start_idx)

    fwd_dist_to_end = forward.get(end_idx, float('inf'))
    bwd_dist_to_end = backward.get(end_idx, float('inf'))

    if fwd_dist_to_end <= bwd_dist_to_end:
        arc_dict, total_dist = forward, fwd_dist_to_end
    else:
        arc_dict, total_dist = backward, bwd_dist_to_end

    path_points = [(idx, d) for idx, d in arc_dict.items() if d <= total_dist]
    path_points.sort(key=lambda p: p[1])

    if total_dist == 0 or len(path_points) < 2:
        return [start_point] * num_samples

    target_arcs = [total_dist * i / (num_samples - 1) for i in range(num_samples)]
    sampled = []
    for target_arc in target_arcs:
        closest = min(path_points, key=lambda p: abs(p[1] - target_arc))
        pt = contour[closest[0]][0]
        sampled.append((int(pt[0]), int(pt[1])))

    return sampled