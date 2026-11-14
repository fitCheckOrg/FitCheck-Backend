"""
ONB-4 E: Discover "For you" (2026-11-14). Work lives in pipelines/for_you_pipeline.py.

GET /api/social/feed/for-you?limit=20&offset=0&category=
  -> {outfits, pagination {total, limit, offset, has_more}, tuned_to?, thin?}
Same post shape as GET /api/social/feed. tuned_to + thin only on the first page
(offset 0, no category) and only when the feed was personalised.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, Query

from shared.dependencies import get_current_user_id
from shared.models.enums import DiscoverCategory
from shared.models.responses import SuccessResponse
from pipelines.for_you_pipeline import for_you_feed

router = APIRouter(prefix="/social", tags=["Social"])


@router.get("/feed/for-you", response_model=SuccessResponse)
def get_for_you(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    category: DiscoverCategory | None = Query(None),
    user_id: UUID = Depends(get_current_user_id),
):
    return SuccessResponse(data=for_you_feed(user_id, limit, offset,
                                             category.value if category else None))