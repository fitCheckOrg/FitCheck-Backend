"""
Try-On API - V2.

Thin routes. Real work lives in pipelines/tryon_pipeline.py and
pipelines/tryon_looks_pipeline.py.

POST /api/tryon          body {"garment_id": uuid}
POST /api/tryon/outfit   body {"garment_ids": [top_uuid, bottom_uuid]}   (chained render)
  200  {"success": true, "data": {"status": "completed", "url", "key", "cached", ...}}
  202  {"success": true, "data": {"status": "processing", ...}}  -> retry in ~3 s
  400  TRYON_OUTFIT_UNSUPPORTED (outfit route: not exactly one top + one bottom)
  404  avatar / garment not found
  409  GARMENT_UNSUPPORTED   (shoes, bags, accessories, unknown category)
  502  TRYON_RENDER_ERROR    (model provider failed; next POST retries)

Saved looks (1+ garments + a try-on render; outfits keep their 2-item rule):
POST   /api/tryon/looks                 {"garment_ids", "tryon_key", "name"?}  201 new / 200 existing
GET    /api/tryon/looks?limit&offset    newest first
PATCH  /api/tryon/looks/{look_id}       {"name": "..." | null}
DELETE /api/tryon/looks/{look_id}
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel

from shared.dependencies import get_current_user_id
from shared.models.responses import SuccessResponse
from pipelines.tryon_pipeline import run_tryon_pipeline, run_outfit_tryon_pipeline
from pipelines.tryon_looks_pipeline import save_look, list_looks, delete_look, rename_look

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/tryon",
    tags=["Try-On"],
)


class TryOnRequest(BaseModel):
    garment_id: UUID


class OutfitTryOnRequest(BaseModel):
    garment_ids: list[UUID]


class SaveLookRequest(BaseModel):
    garment_ids: list[UUID]
    tryon_key: str
    name: str | None = None


class RenameLookRequest(BaseModel):
    name: str | None = None


@router.get("/health")
async def health():
    return {"service": "tryon", "version": "2.2.0", "status": "healthy"}


@router.post("")
async def try_on(
    body: TryOnRequest,
    response: Response,
    user_id: UUID = Depends(get_current_user_id),
):
    logger.info("Try-on request - user=%s garment=%s", user_id, body.garment_id)

    result = await run_tryon_pipeline(
        garment_id=body.garment_id,
        user_id=user_id,
    )

    if result.get("status") == "processing":
        response.status_code = 202

    return SuccessResponse(data=result)


@router.post("/outfit")
async def try_on_outfit(
    body: OutfitTryOnRequest,
    response: Response,
    user_id: UUID = Depends(get_current_user_id),
):
    logger.info("Outfit try-on request - user=%s garments=%s", user_id, body.garment_ids)

    result = await run_outfit_tryon_pipeline(
        garment_ids=body.garment_ids,
        user_id=user_id,
    )

    if result.get("status") == "processing":
        response.status_code = 202

    return SuccessResponse(data=result)


@router.post("/looks")
async def save_tryon_look(
    body: SaveLookRequest,
    response: Response,
    user_id: UUID = Depends(get_current_user_id),
):
    look, created = save_look(user_id, body.garment_ids, body.tryon_key, body.name)
    response.status_code = 201 if created else 200
    return SuccessResponse(data=look)


@router.get("/looks")
async def get_tryon_looks(
    limit: int = 50,
    offset: int = 0,
    user_id: UUID = Depends(get_current_user_id),
):
    return SuccessResponse(data=list_looks(user_id, limit, offset))


@router.patch("/looks/{look_id}")
async def rename_tryon_look(
    look_id: UUID,
    body: RenameLookRequest,
    user_id: UUID = Depends(get_current_user_id),
):
    return SuccessResponse(data=rename_look(user_id, look_id, body.name))


@router.delete("/looks/{look_id}")
async def delete_tryon_look(
    look_id: UUID,
    user_id: UUID = Depends(get_current_user_id),
):
    return SuccessResponse(data=delete_look(user_id, look_id))