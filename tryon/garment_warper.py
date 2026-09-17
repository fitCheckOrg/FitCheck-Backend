"""
V1.1 — independent X/Y scaling, testing whether decoupling width
and height scale factors resolves the sticker appearance before
reaching for shape deformation.
"""

from PIL import Image


def warp_garment_simple(garment_img, garment_coords, avatar_coords):
    g_left_ua = garment_coords["left_underarm"]
    g_right_ua = garment_coords["right_underarm"]
    g_hem = garment_coords["hem"]

    crop_top = int(garment_coords["anchor"][1])
    crop_bottom = int(g_hem[1])
    crop_left = int(min(g_left_ua[0], g_right_ua[0]))
    crop_right = int(max(g_left_ua[0], g_right_ua[0]))

    torso_crop = garment_img.crop((crop_left, crop_top, crop_right, crop_bottom))

    # INDEPENDENT X/Y scale — the only change from the previous version
    x_scale = avatar_coords["width"] / garment_coords["width"]
    y_scale = avatar_coords["height"] / garment_coords["height"]

    new_width = int(torso_crop.width * x_scale)
    new_height = int(torso_crop.height * y_scale)

    resized = torso_crop.resize((new_width, new_height), Image.LANCZOS)
    return resized