import time
from fastapi import APIRouter, Query, Depends
from pydantic import BaseModel
from uuid import UUID
from workers.outfit.explainer import analyze_outfit
from shared.logger import get_logger
from shared.models.responses import SuccessResponse
from shared.exceptions import ClothingItemNotFoundError, FitCheckException
from core.storage.supabase_client import supabase
from workers.outfit.compatibility import calculate_compatibility
from workers.outfit.validator import validate_outfit_composition
from shared.models.enums import OutfitOccasion
from shared.models.planned_outfit import CreatePlannedOutfitRequest, UpdatePlannedOutfitRequest
from datetime import date
from shared.dependencies import get_current_user_id
from shared.models.enums import DiscoverCategory



logger = get_logger(__name__)

router = APIRouter(prefix="/outfit", tags=["Outfit"])

MAX_OUTFIT_ITEMS = 10


class CompatibilityRequest(BaseModel):
    item_ids: list[UUID]


class AnalyzeRequest(BaseModel):
    item_ids: list[UUID]
    occasion: OutfitOccasion = OutfitOccasion.EVERYDAY


class SaveOutfitRequest(BaseModel):
    item_ids: list[UUID]
    occasion: OutfitOccasion = OutfitOccasion.EVERYDAY
    name: str | None = None
    category: DiscoverCategory | None = None

class PublishOutfitRequest(BaseModel):
    is_public: bool


# -- Optional personalisation inputs ------------------------------------------
# A user who skipped onboarding (no style profile) or has no avatar must still
# be able to analyze and save outfits. These never raise.

def _style_profile_for(user_id: UUID) -> dict:
    try:
        res = supabase.table("style_profiles")\
            .select("styles, preferred_colors, disliked_colors")\
            .eq("user_id", str(user_id))\
            .maybe_single()\
            .execute()
        return (res.data or {}) if res else {}
    except Exception as exc:
        logger.warning("Style profile unavailable for user=%s: %s", user_id, exc)
        return {}


def _body_profile_for(user_id: UUID) -> dict:
    try:
        res = supabase.table("avatars")\
            .select("body_profile")\
            .eq("user_id", str(user_id))\
            .maybe_single()\
            .execute()
        return ((res.data or {}).get("body_profile") or {}) if res else {}
    except Exception as exc:
        logger.warning("Body profile unavailable for user=%s: %s", user_id, exc)
        return {}


@router.get("/health")
async def health():

    return { "service": "outfit", "version": "1.1.0", "status": "healthy" }


@router.post("/compatibility", response_model=SuccessResponse)
async def check_compatibility(
    body: CompatibilityRequest,
    user_id: UUID = Depends(get_current_user_id)
):
    """
    Instant rule-based compatibility check.
    No AI call - called on every swipe/selection change.
    """
    if len(body.item_ids) > MAX_OUTFIT_ITEMS:
        raise FitCheckException(
            f"An outfit can contain at most {MAX_OUTFIT_ITEMS} items.",
            code="TOO_MANY_ITEMS",
            status_code=400
        )

    if len(body.item_ids) < 2:
        return SuccessResponse(data={
            "overall_score": 100,
            "overall_grade": "A+",
            "color_score": 100,
            "style_score": 100,
            "occasion_score": 100,
            "season_score": 100,
            "strengths": [],
            "warnings": [],
            "message": "Add more items to see compatibility",
            "engine": "rule-based",
            "engine_version": "1.0"
        })

    logger.info(f"Compatibility check - user={user_id} items={len(body.item_ids)}")

    ids = [str(i) for i in body.item_ids]

    result = supabase.table("closet_items")\
        .select("item_id, category, dominant_hex, style, occasions, seasons")\
        .eq("user_id", str(user_id))\
        .in_("item_id", ids)\
        .execute()

    if not result.data or len(result.data) != len(ids):
        raise ClothingItemNotFoundError()

    start = time.perf_counter()
    compatibility = calculate_compatibility(result.data)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("Compatibility calculated in %.2f ms", elapsed_ms)

    compatibility["message"] = None
    compatibility["engine"] = "rule-based"
    compatibility["engine_version"] = "1.0"

    return SuccessResponse(data=compatibility)

@router.post("/analyze", response_model=SuccessResponse)
async def analyze_outfit_endpoint(
    body: AnalyzeRequest,
    user_id: UUID = Depends(get_current_user_id)
):
    """
    Deep AI analysis - called after user pauses on an outfit.
    Does NOT save anything. Uses style profile + avatar body profile
    when they exist; works without them.
    """
    if len(body.item_ids) > MAX_OUTFIT_ITEMS:
        raise FitCheckException(
            f"An outfit can contain at most {MAX_OUTFIT_ITEMS} items.",
            code="TOO_MANY_ITEMS",
            status_code=400
        )

    if len(body.item_ids) < 2:
        raise FitCheckException(
            "Select at least 2 items to analyze an outfit.",
            code="TOO_FEW_ITEMS",
            status_code=400
        )

    logger.info(f"Outfit analysis - user={user_id} items={len(body.item_ids)}")

    ids = [str(i) for i in body.item_ids]

    # Fetch items - richer field set for AI context
    items_result = supabase.table("closet_items")\
        .select("item_id, category, item_type, dominant_color, dominant_hex, "
                "pattern, style, material, occasions, seasons")\
        .eq("user_id", str(user_id))\
        .in_("item_id", ids)\
        .execute()

    if not items_result.data or len(items_result.data) != len(ids):
        raise ClothingItemNotFoundError()

    items = items_result.data
    validate_outfit_composition(items)
    # Rule-based scores first - feeds into the AI prompt
    rule_scores = calculate_compatibility(items)

    ai_result = analyze_outfit(
        items=items,
        style_profile=_style_profile_for(user_id),
        body_profile=_body_profile_for(user_id),
        rule_scores=rule_scores,
        occasion=body.occasion.value
    )

    return SuccessResponse(data={
        **rule_scores,
        "ai": ai_result
    })

@router.post("/save", response_model=SuccessResponse)
async def save_outfit(
    body: SaveOutfitRequest,
    user_id: UUID = Depends(get_current_user_id)
):
    """
    Save a liked outfit combo.
    Re-runs compatibility + AI analysis and persists the result.
    """
    if len(body.item_ids) > MAX_OUTFIT_ITEMS:
        raise FitCheckException(
            f"An outfit can contain at most {MAX_OUTFIT_ITEMS} items.",
            code="TOO_MANY_ITEMS", status_code=400
        )

    ids = [str(i) for i in body.item_ids]
    logger.info(f"Save outfit - user={user_id} items={len(ids)}")

    items_result = supabase.table("closet_items")\
        .select("item_id, category, item_type, dominant_color, dominant_hex, "
                "pattern, style, material, occasions, seasons")\
        .eq("user_id", str(user_id))\
        .in_("item_id", ids)\
        .execute()

    if not items_result.data or len(items_result.data) != len(ids):
        raise ClothingItemNotFoundError()

    items = items_result.data
    validate_outfit_composition(items)
    rule_scores = calculate_compatibility(items)

    ai_result = analyze_outfit(
        items=items, style_profile=_style_profile_for(user_id),
        body_profile=_body_profile_for(user_id), rule_scores=rule_scores,
        occasion=body.occasion.value
    )

    data = {
        "user_id": str(user_id),
        "item_ids": ids,
        "overall_score": rule_scores["overall_score"],
        "overall_grade": rule_scores["overall_grade"],
        "color_score": rule_scores["color_score"],
        "style_score": rule_scores["style_score"],
        "occasion_score": rule_scores["occasion_score"],
        "season_score": rule_scores["season_score"],
        "ai_summary": ai_result["summary"],
        "ai_highlights": ai_result["highlights"],
        "ai_suggestions": ai_result["suggestions"],
        "ai_personalization_note": ai_result["personalization_note"],
        "ai_confidence": ai_result["confidence"],
        "occasion": body.occasion.value,
        "name": body.name,
        "category": body.category.value if body.category else None,
    }

    result = supabase.table("outfits").insert(data).execute()

    if not result.data:
        raise FitCheckException(
            "Failed to save outfit.", code="DATABASE_ERROR", status_code=500
        )

    logger.info(f"Outfit saved - id={result.data[0]['outfit_id']}")
    return SuccessResponse(data=result.data[0])


@router.get("/", response_model=SuccessResponse)
async def get_outfits(
    user_id: UUID = Depends(get_current_user_id),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """List saved outfits for a user"""
    logger.info(f"Get outfits - user={user_id}")

    result = supabase.table("outfits")\
        .select("*", count="exact")\
        .eq("user_id", str(user_id))\
        .order("created_at", desc=True)\
        .range(offset, offset + limit - 1)\
        .execute()

    total = result.count or 0
    return SuccessResponse(data={
        "outfits": result.data,
        "pagination": {
            "total": total, "limit": limit, "offset": offset,
            "has_more": (offset + limit) < total
        }
    })

@router.delete("/{outfit_id}", response_model=SuccessResponse)
async def delete_outfit(outfit_id: UUID, user_id: UUID = Depends(get_current_user_id)):
    """Delete a saved outfit"""
    logger.info(f"Delete outfit - id={outfit_id} user={user_id}")

    result = supabase.table("outfits")\
        .delete()\
        .eq("outfit_id", str(outfit_id))\
        .eq("user_id", str(user_id))\
        .execute()

    if not result.data:
        raise FitCheckException(
            "Outfit not found.", code="OUTFIT_NOT_FOUND", status_code=404
        )

    return SuccessResponse(data={"message": "Outfit deleted successfully"})


@router.post("/planner", response_model=SuccessResponse)
async def create_planned_outfit(
    body: CreatePlannedOutfitRequest,
    user_id: UUID = Depends(get_current_user_id)
):
    """
    Schedule an outfit for a specific date. Upserts on
    (user_id, scheduled_date) - replanning a day overwrites
    the previous entry rather than creating a duplicate.
    """
    logger.info(
        f"Plan outfit - user={user_id} date={body.scheduled_date}"
    )

    ids = [str(i) for i in body.item_ids]

    items_result = supabase.table("closet_items")\
        .select("item_id")\
        .eq("user_id", str(user_id))\
        .in_("item_id", ids)\
        .execute()

    if not items_result.data or len(items_result.data) != len(ids):
        raise ClothingItemNotFoundError()

    data = {
        "user_id": str(user_id),
        "outfit_id": str(body.outfit_id) if body.outfit_id else None,
        "item_ids": ids,
        "scheduled_date": body.scheduled_date.isoformat(),
        "occasion": body.occasion,
        "notes": body.notes,
        "source": body.source.value,
    }

    result = supabase.table("planned_outfits")\
        .upsert(data, on_conflict="user_id,scheduled_date")\
        .execute()

    if not result.data:
        raise FitCheckException(
            "Failed to plan outfit.", code="DATABASE_ERROR", status_code=500
        )

    logger.info(f"Outfit planned - date={body.scheduled_date}")
    return SuccessResponse(data=result.data[0])


@router.get("/planner", response_model=SuccessResponse)
async def get_planned_outfits(
    user_id: UUID = Depends(get_current_user_id),
    start_date: date = Query(...),
    end_date: date = Query(...)
):
    """Get planned outfits within a date range - e.g. one week."""
    logger.info(
        f"Get planner - user={user_id} range={start_date} to {end_date}"
    )

    result = supabase.table("planned_outfits")\
        .select("*")\
        .eq("user_id", str(user_id))\
        .gte("scheduled_date", start_date.isoformat())\
        .lte("scheduled_date", end_date.isoformat())\
        .order("scheduled_date")\
        .execute()

    return SuccessResponse(data={"planned_outfits": result.data})



@router.patch("/{outfit_id}/publish", response_model=SuccessResponse)
async def toggle_outfit_publish(
    outfit_id: UUID,
    body: PublishOutfitRequest,
    user_id: UUID = Depends(get_current_user_id)
):
    """Publish or unpublish an outfit to the public Discover feed."""
    logger.info(
        f"Toggle outfit publish - outfit={outfit_id} user={user_id} is_public={body.is_public}"
    )

    result = supabase.table("outfits")\
        .update({"is_public": body.is_public})\
        .eq("outfit_id", str(outfit_id))\
        .eq("user_id", str(user_id))\
        .execute()

    if not result.data:
        raise FitCheckException(
            "Outfit not found.", code="OUTFIT_NOT_FOUND", status_code=404
        )

    return SuccessResponse(data={
        "outfit_id": str(outfit_id),
        "is_public": result.data[0]["is_public"]
    })

@router.patch("/planner/{planned_id}", response_model=SuccessResponse)
async def update_planned_outfit(
    planned_id: UUID,
    body: UpdatePlannedOutfitRequest,
    user_id: UUID = Depends(get_current_user_id)
):
    """Edit a planned outfit - change items, mark as worn/skipped, etc."""
    updates = {}
    if body.item_ids is not None:
        updates["item_ids"] = [str(i) for i in body.item_ids]
    if body.occasion is not None:
        updates["occasion"] = body.occasion
    if body.notes is not None:
        updates["notes"] = body.notes
    if body.status is not None:
        updates["status"] = body.status.value

    logger.info(
        f"Update planned outfit={planned_id} user={user_id} fields={list(updates.keys())}"
    )

    if not updates:
        return SuccessResponse(data={"message": "Nothing to update"})

    result = supabase.table("planned_outfits")\
        .update(updates)\
        .eq("planned_id", str(planned_id))\
        .eq("user_id", str(user_id))\
        .execute()

    if not result.data:
        raise FitCheckException(
            "Planned outfit not found.", code="PLANNED_OUTFIT_NOT_FOUND", status_code=404
        )

    return SuccessResponse(data=result.data[0])


@router.delete("/planner/{planned_id}", response_model=SuccessResponse)
async def delete_planned_outfit(planned_id: UUID, user_id: UUID = Depends(get_current_user_id)):
    """Remove a planned outfit"""
    logger.info(f"Delete planned outfit={planned_id} user={user_id}")

    result = supabase.table("planned_outfits")\
        .delete()\
        .eq("planned_id", str(planned_id))\
        .eq("user_id", str(user_id))\
        .execute()

    if not result.data:
        raise FitCheckException(
            "Planned outfit not found.", code="PLANNED_OUTFIT_NOT_FOUND", status_code=404
        )

    return SuccessResponse(data={"message": "Planned outfit removed"})