"""
Comments API (C13, 2026-11-03). Thin routes - work lives in pipelines/comments_pipeline.py.

GET    /api/social/posts/{post_id}/comments?sort=relevant|newest&cursor=&limit=
         -> {items: [comment], next_cursor, count, comments_enabled}
POST   /api/social/posts/{post_id}/comments  {text, parent_id?}  -> 201 comment
GET    /api/comments/{comment_id}/replies?cursor=&limit=         -> {items, next_cursor}
PATCH  /api/comments/{comment_id}  {text?} (author, 15 min) | {pinned?} (post owner)
DELETE /api/comments/{comment_id}  (author or post owner, soft)  -> {deleted, comment_id}
POST/DELETE /api/comments/{comment_id}/like                      -> {liked, like_count, comment_id}
POST   /api/comments/{comment_id}/report  {reason, note?}        -> {reported: true}

comment = {comment_id, post_id, parent_id, user, text, mentions, created_at, edited,
           like_count, liked_by_me, reply_count, replies, is_creator, pinned, deleted}

Errors: 404 POST_NOT_FOUND | 404 COMMENT_NOT_FOUND | 400 COMMENT_EMPTY |
        400 COMMENT_TOO_LONG | 403 COMMENTS_OFF | 403 NOT_ALLOWED |
        403 EDIT_WINDOW_CLOSED | 429 RATE_LIMITED (+ Retry-After) | 400 INVALID_CURSOR
"""

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response
from pydantic import BaseModel

from shared.dependencies import get_current_user_id
from shared.models.responses import SuccessResponse
from pipelines.comments_pipeline import (
    list_comments, list_replies, create_comment, edit_comment, delete_comment,
    like_comment, unlike_comment,
)
from pipelines.safety_pipeline import report_comment

router = APIRouter(prefix="/comments", tags=["Comments"])
post_router = APIRouter(prefix="/social", tags=["Comments"])


class NewCommentRequest(BaseModel):
    text: str | None = None
    parent_id: UUID | None = None


class EditCommentRequest(BaseModel):
    text: str | None = None
    pinned: bool | None = None


class ReportRequest(BaseModel):
    reason: str
    note: str | None = None


@router.get("/health")
def health():
    return {"service": "comments", "version": "1.0.0", "status": "healthy"}


@post_router.get("/posts/{post_id}/comments")
def get_post_comments(
    post_id: UUID,
    sort: Literal["relevant", "newest"] = Query("relevant"),
    cursor: str | None = Query(None),
    limit: int = Query(20, ge=1, le=50),
    user_id: UUID = Depends(get_current_user_id),
):
    return SuccessResponse(data=list_comments(post_id, user_id, sort, cursor, limit))


@post_router.post("/posts/{post_id}/comments")
def post_comment(
    post_id: UUID,
    body: NewCommentRequest,
    response: Response,
    user_id: UUID = Depends(get_current_user_id),
):
    response.status_code = 201
    return SuccessResponse(data=create_comment(post_id, user_id, body.text, body.parent_id))


@router.get("/{comment_id}/replies")
def get_replies(
    comment_id: UUID,
    cursor: str | None = Query(None),
    limit: int = Query(20, ge=1, le=50),
    user_id: UUID = Depends(get_current_user_id),
):
    return SuccessResponse(data=list_replies(comment_id, user_id, cursor, limit))


@router.patch("/{comment_id}")
def patch_comment(comment_id: UUID, body: EditCommentRequest,
                  user_id: UUID = Depends(get_current_user_id)):
    sent = {k: getattr(body, k) for k in body.model_fields_set}
    return SuccessResponse(data=edit_comment(comment_id, user_id, sent))


@router.delete("/{comment_id}")
def remove_comment(comment_id: UUID, user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=delete_comment(comment_id, user_id))


@router.post("/{comment_id}/like")
def post_like(comment_id: UUID, user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=like_comment(comment_id, user_id))


@router.delete("/{comment_id}/like")
def delete_like(comment_id: UUID, user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=unlike_comment(comment_id, user_id))


@router.post("/{comment_id}/report")
def post_report_comment(comment_id: UUID, body: ReportRequest,
                        user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=report_comment(user_id, comment_id, body.reason, body.note))