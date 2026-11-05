"""
Activity API (D15, 2026-11-03) - the bell. Work lives in pipelines/activity_pipeline.py.

GET  /api/social/activity?cursor=&limit=   -> {items: [activity], next_cursor}
GET  /api/social/activity/unread-count     -> {unread_count}
POST /api/social/activity/read  {up_to_activity_id?}  -> {unread_count, marked}

activity = {activity_id, type, actors: [user, user], actor_count,
            post: {post_id, items: [first 3 pieces]} | null,
            comment: {comment_id, text_preview} | null,
            reaction_type, milestone: {kind, value} | null, created_at, read}
"""

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from shared.dependencies import get_current_user_id
from shared.models.responses import SuccessResponse
from pipelines.activity_pipeline import list_activity, unread_count, mark_read

router = APIRouter(prefix="/social", tags=["Activity"])


class MarkReadRequest(BaseModel):
    up_to_activity_id: UUID | None = None


@router.get("/activity/health")
def health():
    return {"service": "activity", "version": "1.0.0", "status": "healthy"}


@router.get("/activity")
def get_activity(
    cursor: str | None = Query(None),
    limit: int = Query(20, ge=1, le=50),
    user_id: UUID = Depends(get_current_user_id),
):
    return SuccessResponse(data=list_activity(user_id, cursor, limit))


@router.get("/activity/unread-count")
def get_unread_count(user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=unread_count(user_id))


@router.post("/activity/read")
def post_read(body: MarkReadRequest | None = None, user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=mark_read(user_id, body.up_to_activity_id if body else None))