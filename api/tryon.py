"""
Try-On API - V2.

Thin routes. Real work lives in the pipelines below.

Try-On v2 T2-1 B7b (2026-11-25): the old blocking routes POST /api/tryon and
POST /api/tryon/outfit are RETIRED (the app renders through outfits + jobs since A3).
Saved looks accept the new body {outfit_id, image_id, name?}; the old body still works.

Saved looks (1+ garments + a try-on render):
POST   /api/tryon/looks   {"outfit_id", "image_id", "name"?}          201 new / 200 existing
                          or (old) {"garment_ids", "tryon_key", "name"?}
  400 LOOK_BODY_INVALID | LOOK_GARMENT_MISMATCH | LOOK_NAME_TOO_LONG | ...
  404 OUTFIT_NOT_FOUND | TRYON_RENDER_NOT_FOUND
GET    /api/tryon/looks?limit&offset    newest first
PATCH  /api/tryon/looks/{look_id}       {"name": "..." | null}
DELETE /api/tryon/looks/{look_id}

Try-On v2 T2-1 B1 (2026-11-17):
GET  /api/tryon/slots                  -> {slots: [{slot, renders, max, excludes, tray_when, categories}],
                                           render_order}
POST /api/tryon/outfits  {items: [{garment_id, slot?, fit?}]}  -> outfit (same content -> same id)
GET  /api/tryon/outfits/{outfit_id}    -> outfit
  outfit = {outfit_id, items: [{slot, position, garment_id, fit, renders, missing}],
            rendered_slots, tray_slots, needs_generations, cached_image, created_at}
  fit (B7b) = {silhouette: slim|regular|relaxed|oversized, length: cropped|regular|long,
               styling: tucked|untucked|front_tuck|layered, rise: low|mid|high}  (all optional)
  400 OUTFIT_EMPTY | SLOT_INVALID | SLOT_CONFLICT | FIT_INVALID
  404 OUTFIT_NOT_FOUND | CLOTHING_ITEM_NOT_FOUND

Try-On v2 T2-1 B2 (2026-11-17) - render jobs (rendered in the background, see
pipelines/render_worker.py):
POST /api/tryon/jobs  {outfit_id} | {items: [...]}   -> 202 new job / 200 already rendered or on its way
GET  /api/tryon/jobs/{job_id}                         -> job
GET  /api/tryon/jobs?active=true                      -> {items: [job]}  (queued + running)
POST /api/tryon/jobs/{job_id}/cancel                  -> job
  job = {job_id, outfit_id, status, progress {step, steps}, queue_position, rendered_slots,
         tray_slots, charged_generations, provider_started, failure {code, message} | null,
         image | null, created_at, finished_at}
  400 NOTHING_TO_RENDER | OUTFIT_EMPTY   404 JOB_NOT_FOUND | AVATAR_NOT_FOUND
  429 RATE_LIMITED (2 renders on the way already)

Try-On v2 T2-1 B3 (2026-11-17) - quotas:
GET  /api/tryon/usage   -> {period, used, reserved, limit, remaining, resets_at, tier}
POST /api/tryon/jobs    also: 429 QUOTA_EXCEEDED {used, reserved, limit, needed, resets_at}
                              429 RATE_LIMITED {retry_after} (2 active, or 30 creates / hour)
                              503 RENDERS_PAUSED {retry_after} (daily ceiling, everyone)
  Already-rendered outfits still open (200) when out of renders or paused.
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, Field

from shared.dependencies import get_current_user_id
from shared.models.responses import SuccessResponse
from pipelines.tryon_looks_pipeline import save_look, list_looks, delete_look, rename_look
from pipelines.tryon_slots import slots_payload
from pipelines.tryon_outfits_pipeline import create_outfit, get_outfit
from pipelines.render_jobs_pipeline import create_job, get_job, list_active, cancel_job, get_usage

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/tryon",
    tags=["Try-On"],
)


class SaveLookRequest(BaseModel):
    # B7b: new body {outfit_id, image_id, name?}; old body {garment_ids, tryon_key, name?}
    outfit_id: UUID | None = None
    image_id: UUID | None = None
    garment_ids: list[UUID] | None = None
    tryon_key: str | None = None
    name: str | None = None


class RenameLookRequest(BaseModel):
    name: str | None = None


class OutfitItem(BaseModel):                       # T2-1 B1
    garment_id: UUID
    slot: str | None = Field(default=None, max_length=20)
    fit: dict | None = None


class CreateOutfitRequest(BaseModel):
    items: list[OutfitItem] = Field(default_factory=list, max_length=20)


class CreateJobRequest(BaseModel):                 # T2-1 B2
    outfit_id: UUID | None = None
    items: list[OutfitItem] | None = Field(default=None, max_length=20)


@router.get("/health")
async def health():
    return {"service": "tryon", "version": "2.6.0", "status": "healthy"}


# ---------------------------------------------------------------- T2-1 B1: slots + outfits

@router.get("/slots")
async def get_slots(user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=slots_payload())


@router.post("/outfits")
def post_outfit(body: CreateOutfitRequest, user_id: UUID = Depends(get_current_user_id)):
    items = [{"garment_id": i.garment_id, "slot": i.slot, "fit": i.fit} for i in body.items]
    return SuccessResponse(data=create_outfit(user_id, items))


@router.get("/outfits/{outfit_id}")
def read_outfit(outfit_id: UUID, user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=get_outfit(user_id, outfit_id))


# ---------------------------------------------------------------- T2-1 B2: render jobs

@router.post("/jobs")
def post_job(body: CreateJobRequest, response: Response,
             user_id: UUID = Depends(get_current_user_id)):
    items = [{"garment_id": i.garment_id, "slot": i.slot, "fit": i.fit} for i in (body.items or [])]
    job, created = create_job(user_id, body.outfit_id, items or None)
    response.status_code = 202 if created else 200
    return SuccessResponse(data=job)


@router.get("/usage")
def read_usage(user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=get_usage(user_id))


@router.get("/jobs")
def read_active_jobs(active: bool = True, user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=list_active(user_id))


@router.get("/jobs/{job_id}")
def read_job(job_id: UUID, user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=get_job(user_id, job_id))


@router.post("/jobs/{job_id}/cancel")
def post_cancel_job(job_id: UUID, user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=cancel_job(user_id, job_id))


# ---------------------------------------------------------------- saved looks

@router.post("/looks")
def save_tryon_look(
    body: SaveLookRequest,
    response: Response,
    user_id: UUID = Depends(get_current_user_id),
):
    look, created = save_look(user_id, garment_ids=body.garment_ids, tryon_key=body.tryon_key,
                              name=body.name, outfit_id=body.outfit_id, image_id=body.image_id)
    response.status_code = 201 if created else 200
    return SuccessResponse(data=look)


@router.get("/looks")
def get_tryon_looks(
    limit: int = 50,
    offset: int = 0,
    user_id: UUID = Depends(get_current_user_id),
):
    return SuccessResponse(data=list_looks(user_id, limit, offset))


@router.patch("/looks/{look_id}")
def rename_tryon_look(
    look_id: UUID,
    body: RenameLookRequest,
    user_id: UUID = Depends(get_current_user_id),
):
    return SuccessResponse(data=rename_look(user_id, look_id, body.name))


@router.delete("/looks/{look_id}")
def delete_tryon_look(
    look_id: UUID,
    user_id: UUID = Depends(get_current_user_id),
):
    return SuccessResponse(data=delete_look(user_id, look_id))
