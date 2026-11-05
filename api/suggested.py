"""
Suggested creators API (N2, 2026-11-03). Work lives in pipelines/suggested_pipeline.py.

GET /api/users/suggested?limit=20
  -> {items: [user + counts {followers, posts} + preview_posts [{post_id, items}] + reason]}
"""

from uuid import UUID

from fastapi import APIRouter, Depends, Query

from shared.dependencies import get_current_user_id
from shared.models.responses import SuccessResponse
from pipelines.suggested_pipeline import suggested_creators

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/suggested/health")
def health():
    return {"service": "suggested", "version": "1.0.0", "status": "healthy"}


@router.get("/suggested")
def get_suggested(
    limit: int = Query(20, ge=1, le=50),
    user_id: UUID = Depends(get_current_user_id),
):
    return SuccessResponse(data=suggested_creators(user_id, limit))