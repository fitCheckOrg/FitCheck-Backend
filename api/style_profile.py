from fastapi import APIRouter, Query, Depends
from uuid import UUID
from shared.logger import get_logger
from shared.models.responses import SuccessResponse
from shared.models.style_profile import StyleProfile, UpdateStyleProfileRequest
from shared.exceptions import FitCheckException
from core.storage.supabase_client import supabase
from shared.dependencies import get_current_user_id


logger = get_logger(__name__)

router = APIRouter(prefix="/style-profile", tags=["Style Profile"])


def _empty_profile(user_id: UUID) -> dict:
    return {
        "user_id": str(user_id),
        "styles": [],
        "preferred_colors": [],
        "disliked_colors": [],
        "favorite_occasions": [],
        "self_reported_body_shape": None,
        "onboarding_completed": False
    }


@router.get("/health")
async def health():
    return { "service": "style-profile", "version": "1.1.0", "status": "healthy" }


@router.post("/create", response_model=SuccessResponse)
async def create_style_profile(
    profile: StyleProfile,
    user_id: UUID = Depends(get_current_user_id)
):
    """
    Create or replace style profile - called after onboarding quiz.
    Upsert - one profile per user always. The user comes from the token;
    a user_id in the body is optional and ignored.
    """
    logger.info(f"Create style profile - user={user_id}")

    data = {
        "user_id": str(user_id),
        "styles": profile.styles,
        "preferred_colors": profile.preferred_colors,
        "disliked_colors": profile.disliked_colors,
        "favorite_occasions": profile.favorite_occasions,
        "self_reported_body_shape": profile.self_reported_body_shape,
        "onboarding_completed": True
    }

    result = supabase.table("style_profiles")\
        .upsert(data, on_conflict="user_id")\
        .execute()

    if not result.data:
        raise FitCheckException(
            "Failed to save style profile.",
            code="DATABASE_ERROR",
            status_code=500
        )

    logger.info(f"Style profile saved - user={user_id}")
    return SuccessResponse(data=result.data[0])

@router.get("/me", response_model=SuccessResponse)
async def get_my_style_profile(user_id: UUID = Depends(get_current_user_id)):
    """Get current user's style profile (empty defaults if not onboarded yet)"""
    logger.info(f"Get style profile - user={user_id}")

    result = supabase.table("style_profiles")\
        .select("*")\
        .eq("user_id", str(user_id))\
        .maybe_single()\
        .execute()

    if not result or not result.data:
        return SuccessResponse(data=_empty_profile(user_id))

    return SuccessResponse(data=result.data)


@router.patch("/me", response_model=SuccessResponse)
async def update_style_profile(
    body: UpdateStyleProfileRequest,
    user_id: UUID = Depends(get_current_user_id)
):
    """Edit style profile - partial update"""
    updates = body.model_dump(exclude_none=True, exclude_unset=True)

    logger.info(
        "Update style profile - user=%s fields=%s",
        user_id,
        list(updates.keys())
    )

    if not updates:
        return SuccessResponse(data={"message": "Nothing to update"})

    result = supabase.table("style_profiles")\
        .update(updates)\
        .eq("user_id", str(user_id))\
        .execute()

    if not result.data:
        raise FitCheckException(
            "Style profile not found. Please complete onboarding first.",
            code="STYLE_PROFILE_NOT_FOUND",
            status_code=404
        )

    return SuccessResponse(data=result.data[0])