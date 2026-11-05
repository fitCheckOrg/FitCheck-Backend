"""
Related posts API (D16, 2026-11-03). Work lives in pipelines/related_pipeline.py.

GET /api/social/posts/{post_id}/related?limit=12   -> {items: [feed item]}
Errors: 404 POST_NOT_FOUND
"""

from uuid import UUID

from fastapi import APIRouter, Depends, Query

from shared.dependencies import get_current_user_id
from shared.models.responses import SuccessResponse
from pipelines.related_pipeline import related_posts

router = APIRouter(prefix="/social", tags=["Related"])


@router.get("/related/health")
def health():
    return {"service": "related", "version": "1.0.0", "status": "healthy"}


@router.get("/posts/{post_id}/related")
def get_related(
    post_id: UUID,
    limit: int = Query(12, ge=1, le=30),
    user_id: UUID = Depends(get_current_user_id),
):
    return SuccessResponse(data=related_posts(post_id, user_id, limit))