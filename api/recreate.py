"""
Recreate API (D17, 2026-11-03). Work lives in pipelines/recreate_pipeline.py.

POST /api/stylist/recreate {"post_id"}
  -> {post_id, items, pieces, matched, total, missing, overall_score, overall_grade}
Errors: 404 POST_NOT_FOUND | 400 INSUFFICIENT_WARDROBE
"""

from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from shared.dependencies import get_current_user_id
from shared.models.responses import SuccessResponse
from pipelines.recreate_pipeline import recreate

router = APIRouter(prefix="/stylist", tags=["Recreate"])


class RecreateRequest(BaseModel):
    post_id: UUID


@router.get("/recreate/health")
def health():
    return {"service": "recreate", "version": "1.0.0", "status": "healthy"}


@router.post("/recreate")
def post_recreate(body: RecreateRequest, user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=recreate(body.post_id, user_id))