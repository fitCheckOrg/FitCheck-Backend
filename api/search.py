"""
Search API (D14, 2026-11-03). Work lives in pipelines/search_pipeline.py.

GET /api/social/search?q=&type=posts|creators|styles&cursor=&limit=
      posts    -> {items: [feed item], next_cursor}
      creators -> {items: [user + counts {followers}], next_cursor}
      styles   -> {items: [{tag, label, post_count}], next_cursor}
GET /api/social/search/suggestions -> {trending_styles: [{tag, label}], colours: [{name, hex}]}

Errors: 400 QUERY_TOO_SHORT (< 2 characters) | 400 INVALID_CURSOR
"""

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query

from shared.dependencies import get_current_user_id
from shared.models.responses import SuccessResponse
from pipelines.search_pipeline import search, suggestions

router = APIRouter(prefix="/social", tags=["Search"])


@router.get("/search/health")
def health():
    return {"service": "search", "version": "1.0.0", "status": "healthy"}


@router.get("/search/suggestions")
def get_suggestions(user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=suggestions())


@router.get("/search")
def get_search(
    q: str = Query(..., max_length=100),
    type: Literal["posts", "creators", "styles"] = Query("posts"),
    cursor: str | None = Query(None),
    limit: int = Query(20, ge=1, le=50),
    user_id: UUID = Depends(get_current_user_id),
):
    return SuccessResponse(data=search(q, type, user_id, cursor, limit))