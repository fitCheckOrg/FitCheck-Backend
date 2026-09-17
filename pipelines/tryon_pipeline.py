"""
Try-on rendering pipeline.

V1:
    avatar + garment
        → representations
        → fitting
        → compositing
        → S3 render
        → URL/key

No database persistence.
"""

import io
import logging
from uuid import UUID

from core.storage.s3 import storage, S3Folders

from tryon.avatar_representation import build_avatar_representation
from tryon.garment_representation import build_garment_representation
from tryon.fitting_engine import FittingEngine
from tryon.compositor import Compositor
from tryon.models import TryOnResult

from shared.exceptions import FitCheckException, GarmentUnsupportedError

logger = logging.getLogger(__name__)


async def run_tryon_pipeline(
    garment_id: UUID,
    user_id: UUID,
) -> TryOnResult:
    """
    Execute the V1 try-on pipeline.

    Does not persist a database record — only uploads the rendered
    image to S3 and returns its URL and storage key.
    """
    try:
        # 1. Obtain representations — both are sync functions, called
        # directly (matches existing project convention: async pipeline
        # functions call sync helpers without threadpool wrapping)
        avatar = build_avatar_representation(str(user_id))
        garment = build_garment_representation(str(garment_id), str(user_id))

        # 2. V1 capability gate
        if not garment.shoulders_available:
            raise GarmentUnsupportedError()

        # 3. Fit garment to avatar
        fitting_result = FittingEngine.fit(garment, avatar)

        # 4. Composite fitted layers over avatar
        rendered_image = Compositor.composite(avatar.image, fitting_result)

        # 5. PIL Image → PNG bytes (matches enhancer.py's exact pattern)
        output = io.BytesIO()
        rendered_image.save(output, format="PNG", optimize=True, compress_level=6)
        output.seek(0)
        image_bytes = output.read()

        # 6. Store rendered result
        uploaded = storage.upload(
            image_bytes,
            folder=S3Folders.TRYON,
            extension="png",
            content_type="image/png",
        )

        # 7. Return render reference
        return TryOnResult(url=uploaded["url"], key=uploaded["key"])

    except FitCheckException:
        raise

    except Exception as exc:
        logger.exception("Try-on pipeline failed — user=%s garment=%s", user_id, garment_id)
        raise FitCheckException(
            "Try-on rendering failed.",
            code="TRYON_RENDER_ERROR",
            status_code=500,
        ) from exc