"""
Insights API (E19, 2026-11-03). Work lives in pipelines/insights_pipeline.py.

POST /api/social/posts/views        {"post_ids": [...]}            -> {ok, counted}
POST /api/social/posts/{id}/share   {"channel": image|system|link} -> {post_id, share_count}
GET  /api/social/posts/{id}/insights?range=7d|30d                  -> insights (owner only)

Errors: 404 POST_NOT_FOUND | 403 NOT_OWNER | 400 INVALID_CHANNEL | 400 INVALID_RANGE

IMPORTANT: include this router BEFORE the social router in main.py, so
POST /social/posts/views isn't taken by POST /social/posts/{post_id}.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from shared.dependencies import get_current_user_id
from shared.models.responses import SuccessResponse
from pipelines.insights_pipeline import record_views, record_share, post_insights

router = APIRouter(prefix="/social", tags=["Insights"])


class ViewsRequest(BaseModel):
    post_ids: list[UUID] = Field(default_factory=list, max_length=100)


class ShareRequest(BaseModel):
    channel: str


@router.get("/insights/health")
def health():
    return {"service": "insights", "version": "1.0.0", "status": "healthy"}


@router.post("/posts/views")
def post_views(body: ViewsRequest, user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=record_views(body.post_ids, user_id))


@router.post("/posts/{post_id}/share")
def post_share(post_id: UUID, body: ShareRequest, user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=record_share(post_id, user_id, body.channel))


@router.get("/posts/{post_id}/insights")
def get_insights(
    post_id: UUID,
    range: str = Query("7d"),
    user_id: UUID = Depends(get_current_user_id),
):
    return SuccessResponse(data=post_insights(post_id, user_id, range))