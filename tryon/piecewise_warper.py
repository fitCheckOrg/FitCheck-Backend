"""
Piecewise affine warp, generalized to N correspondence points per
side via a standard triangle-strip pattern between two parallel
point chains (left side chain, right side chain).
"""

import numpy as np
import cv2
from PIL import Image


def warp_triangle(src_img, src_tri_pts, dst_tri_pts, output_size):
    src_tri = np.array(src_tri_pts, dtype=np.float32).reshape(3, 2)
    dst_tri = np.array(dst_tri_pts, dtype=np.float32).reshape(3, 2)

    src_rect = cv2.boundingRect(np.array(src_tri_pts, dtype=np.int32))
    dst_rect = cv2.boundingRect(np.array(dst_tri_pts, dtype=np.int32))

    if src_rect[2] <= 0 or src_rect[3] <= 0 or dst_rect[2] <= 0 or dst_rect[3] <= 0:
        return None, dst_rect

    src_tri_offset = np.array(
        [[p[0] - src_rect[0], p[1] - src_rect[1]] for p in src_tri_pts],
        dtype=np.float32
    ).reshape(3, 2)
    dst_tri_offset = np.array(
        [[p[0] - dst_rect[0], p[1] - dst_rect[1]] for p in dst_tri_pts],
        dtype=np.float32
    ).reshape(3, 2)

    src_crop = src_img[src_rect[1]:src_rect[1]+src_rect[3],
                        src_rect[0]:src_rect[0]+src_rect[2]]
    if src_crop.size == 0:
        return None, dst_rect

    warp_mat = cv2.getAffineTransform(src_tri_offset, dst_tri_offset)
    dst_crop = cv2.warpAffine(src_crop, warp_mat, (dst_rect[2], dst_rect[3]),
                                borderMode=cv2.BORDER_REFLECT_101)

    mask = np.zeros((dst_rect[3], dst_rect[2], 4), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_tri_offset.astype(np.int32), (255, 255, 255, 255))

    dst_crop = cv2.bitwise_and(dst_crop, mask)
    return dst_crop, dst_rect


def piecewise_warp_garment_strip(garment_rgba, src_left_chain, src_right_chain,
                                    dst_left_chain, dst_right_chain, output_size):
    """
    Two parallel point chains (left side, right side), same length,
    ordered underarm -> hem. Builds a standard triangle strip
    between them: for each i, triangles (L[i], R[i], L[i+1]) and
    (R[i], L[i+1], R[i+1]).
    """
    assert len(src_left_chain) == len(src_right_chain) == len(dst_left_chain) == len(dst_right_chain)
    n = len(src_left_chain)

    src_arr = np.array(garment_rgba)
    output = np.zeros((output_size[1], output_size[0], 4), dtype=np.uint8)

    for i in range(n - 1):
        pairs = [
            (src_left_chain[i], src_right_chain[i], src_left_chain[i+1],
             dst_left_chain[i], dst_right_chain[i], dst_left_chain[i+1]),
            (src_right_chain[i], src_left_chain[i+1], src_right_chain[i+1],
             dst_right_chain[i], dst_left_chain[i+1], dst_right_chain[i+1]),
        ]
        for src_a, src_b, src_c, dst_a, dst_b, dst_c in pairs:
            src_tri_pts = [src_a, src_b, src_c]
            dst_tri_pts = [dst_a, dst_b, dst_c]

            warped_crop, dst_rect = warp_triangle(src_arr, src_tri_pts, dst_tri_pts, output_size)
            if warped_crop is None:
                continue

            x, y, w, h = dst_rect
            x_end, y_end = min(x + w, output.shape[1]), min(y + h, output.shape[0])
            x_start, y_start = max(x, 0), max(y, 0)
            if x_end <= x_start or y_end <= y_start:
                continue

            crop_x_off = x_start - x
            crop_y_off = y_start - y
            region = output[y_start:y_end, x_start:x_end]
            warped_region = warped_crop[crop_y_off:crop_y_off+(y_end-y_start),
                                         crop_x_off:crop_x_off+(x_end-x_start)]

            alpha = warped_region[:, :, 3:4] / 255.0
            output[y_start:y_end, x_start:x_end] = (
                region * (1 - alpha) + warped_region * alpha
            ).astype(np.uint8)

    return Image.fromarray(output, mode="RGBA")