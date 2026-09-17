"""
Try-On API — V1.

Thin route. All real work lives in pipelines/tryon_pipeline.py.
No database persistence, no history, no queue — matches the
explicitly scoped V1 boundary.
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from shared.dependencies import get_current_user_id
from shared.models.responses import SuccessResponse
from pipelines.tryon_pipeline import run_tryon_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/tryon",
    tags=["Try-On"],
)


class TryOnRequest(BaseModel):
    garment_id: UUID


@router.get("/health")
async def health():
    return {"service": "tryon", "version": "1.0.0", "status": "healthy"}


@router.post("")
async def try_on(
    body: TryOnRequest,
    user_id: UUID = Depends(get_current_user_id),
):
    logger.info("Try-on request — user=%s garment=%s", user_id, body.garment_id)

    result = await run_tryon_pipeline(
        garment_id=body.garment_id,
        user_id=user_id,
    )

    return SuccessResponse(data=result)