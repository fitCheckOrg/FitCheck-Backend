"""
Try-On Studio B14-0 (2026-11-27): sharp garment cut-outs, no AI.

remove.bg answers at its free preview size (~0.25 MP, e.g. 511 x 447). We only need
its OUTLINE (the transparency): it is laid on the ORIGINAL photo's own pixels, up to
MAX_IMAGE_SIZE on the long side, then trimmed to the garment + a small margin.
Every pixel of the garment comes from the user's photo; nothing is invented and no
colour / contrast / sharpness is changed.

    make_cutout(original_bytes, uncropped_cutout_bytes) -> {
        "bytes", "width", "height",
        "bbox": [x, y, w, h],        # visible garment inside the cut-out, fractions 0-1
        "source": "original_pixels" | "removebg"   # removebg = shapes didn't match
    }

remove.bg must be called with crop=False, so its result has the original's framing.
"""

import io

from PIL import Image, ImageOps

from core.config.settings import settings
from shared.logger import get_logger

logger = get_logger(__name__)

RATIO_TOLERANCE = 0.01
ALPHA_VISIBLE = 16          # alpha above this counts as garment when trimming
MARGIN = 0.03               # empty border kept around the garment (of its long side)


class CutoutError(Exception):
    """No garment left after background removal."""


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True, compress_level=6)
    return buf.getvalue()


def make_cutout(original_bytes: bytes, cutout_bytes: bytes) -> dict:
    cut = Image.open(io.BytesIO(cutout_bytes)).convert("RGBA")
    orig = ImageOps.exif_transpose(Image.open(io.BytesIO(original_bytes))).convert("RGB")

    ow, oh = orig.size
    cw, ch = cut.size
    source = "removebg"
    img = cut
    if abs(ow / oh - cw / ch) <= RATIO_TOLERANCE:
        max_side = int(settings.MAX_IMAGE_SIZE)
        orig.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        if orig.size[0] > cw:
            alpha = cut.getchannel("A").resize(orig.size, Image.Resampling.LANCZOS)
            img = orig.convert("RGBA")
            img.putalpha(alpha)
            source = "original_pixels"
    else:
        logger.warning("Cut-out %dx%d doesn't match original %dx%d - using remove.bg's pixels",
                       cw, ch, ow, oh)

    visible = img.getchannel("A").point(lambda a: 255 if a > ALPHA_VISIBLE else 0).getbbox()
    if not visible:
        raise CutoutError("No garment found in the photo after background removal.")
    x0, y0, x1, y1 = visible
    m = round(MARGIN * max(x1 - x0, y1 - y0))
    W, H = img.size
    box = (max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m))
    img = img.crop(box)
    w, h = img.size
    bbox = [round((x0 - box[0]) / w, 4), round((y0 - box[1]) / h, 4),
            round((x1 - x0) / w, 4), round((y1 - y0) / h, 4)]
    logger.info("Cut-out %dx%d from %s (remove.bg gave %dx%d)", w, h, source, cw, ch)
    return {"bytes": _png(img), "width": w, "height": h, "bbox": bbox, "source": source}