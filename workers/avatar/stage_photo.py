"""
Try-On Studio B10b (2026-11-26): the avatar's stage photo (processed_photo_url).

remove.bg sends its cut-out at its free preview size (about 0.25 megapixels, e.g.
375 x 666) - too soft for a phone screen. Since B10 the cut-out has exactly the
original photo's framing, so we keep remove.bg's transparency (its outline of the
person) and put it on the ORIGINAL photo's own pixels:

    stage photo = original pixels (up to MAX_IMAGE_SIZE on the long side)
                  + remove.bg's transparency, scaled up to the same size

* No colour, contrast or brightness changes: the person's skin and face are exactly
  as in their photo, and match the try-on renders (FASHN is given the original too).
* If the cut-out's shape doesn't match the original (e.g. an old cropped cut-out),
  the cut-out is used as it is.
"""

import io

from PIL import Image, ImageOps

from core.config.settings import settings
from shared.logger import get_logger
from workers.avatar.outline import measure_outline
from workers.avatar.pose_estimator import estimate_pose

logger = get_logger(__name__)

RATIO_TOLERANCE = 0.01


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True, compress_level=6)
    return buf.getvalue()


def build_stage_photo(original_bytes: bytes, cutout_bytes: bytes) -> bytes:
    cut = Image.open(io.BytesIO(cutout_bytes)).convert("RGBA")
    orig = ImageOps.exif_transpose(Image.open(io.BytesIO(original_bytes))).convert("RGB")

    ow, oh = orig.size
    cw, ch = cut.size
    if abs(ow / oh - cw / ch) > RATIO_TOLERANCE:
        logger.warning("Stage photo - cut-out %dx%d doesn't match original %dx%d; using the cut-out",
                       cw, ch, ow, oh)
        return _png(cut)

    max_side = int(settings.MAX_IMAGE_SIZE)
    orig.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    if orig.size[0] <= cw:                       # the cut-out is already as big - keep it
        return _png(cut)

    alpha = cut.getchannel("A").resize(orig.size, Image.Resampling.LANCZOS)
    stage = orig.convert("RGBA")
    stage.putalpha(alpha)
    logger.info("Stage photo - %dx%d (cut-out was %dx%d)", stage.size[0], stage.size[1], cw, ch)
    return _png(stage)


def make_stage(original_bytes: bytes, cutout_bytes: bytes) -> dict:
    """Stage photo + its size, body points and outline points - one picture frame."""
    stage = build_stage_photo(original_bytes, cutout_bytes)
    w, h = Image.open(io.BytesIO(stage)).size
    pose = estimate_pose(stage)
    outline = measure_outline(stage, pose) if pose else None
    return {"bytes": stage, "width": w, "height": h,
            "pose_keypoints": pose, "body_outline": outline}