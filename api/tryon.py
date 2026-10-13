"""
Try-On API — V2.

Thin route. All real work lives in pipelines/tryon_pipeline.py.

POST /api/tryon   body {"garment_id": uuid}
  200  {"success": true, "data": {"status": "completed", "url", "key", "cached"}}
  202  {"success": true, "data": {"status": "processing"}}
       -> another request is already rendering this pair; retry in ~3 s
  404  avatar / garment not found
  409  GARMENT_UNSUPPORTED   (shoes, bags, accessories, unknown category)
  502  TRYON_RENDER_ERROR    (model provider failed; next POST retries)
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, Response
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
    return {"service": "tryon", "version": "2.0.0", "status": "healthy"}


@router.post("")
async def try_on(
    body: TryOnRequest,
    response: Response,
    user_id: UUID = Depends(get_current_user_id),
):
    logger.info("Try-on request — user=%s garment=%s", user_id, body.garment_id)

    result = await run_tryon_pipeline(
        garment_id=body.garment_id,
        user_id=user_id,
    )

    if result.get("status") == "processing":
        response.status_code = 202

    return SuccessResponse(data=result)