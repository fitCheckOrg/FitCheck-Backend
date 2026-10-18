from fastapi import APIRouter, Query, Depends, Response
from shared.dependencies import get_current_user_id
from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime, UTC
from shared.logger import get_logger
from shared.models.responses import SuccessResponse
from shared.models.enums import OutfitOccasion, OutfitMood
from pipelines.stylist_pipeline import run_stylist_pipeline
from pipelines.stylist_feedback_pipeline import record_feedback
from workers.stylist.intent_extractor import extract_intent


logger = get_logger(__name__)

router = APIRouter(prefix="/stylist", tags=["Stylist"])


class GenerateRequest(BaseModel):
    occasion: OutfitOccasion = OutfitOccasion.EVERYDAY
    mood: OutfitMood | None = None
    latitude: float | None = None
    longitude: float | None = None
    excluded_item_ids: list[UUID] = Field(default_factory=list)
    max_results: int = 3

class DressMeForThisRequest(BaseModel):
    text: str
    latitude: float | None = None
    longitude: float | None = None
    excluded_item_ids: list[UUID] = Field(default_factory=list)
    max_results: int = 3

class FeedbackRequest(BaseModel):
    action: str
    item_ids: list[UUID]
    occasion: str | None = None
    mood: str | None = None
    source: str | None = None
    replaced_item_id: UUID | None = None
    replacement_item_id: UUID | None = None

@router.get("/health")
async def health():
    return { "service": "stylist", "version": "1.1.0", "status": "healthy" }


@router.post("/generate", response_model=SuccessResponse)
async def generate_outfit_suggestions(
    body: GenerateRequest,
    user_id: UUID = Depends(get_current_user_id)
):
    """
    Fast, instant outfit generation - no AI call.
    Returns top N scored candidates. Call POST /api/outfit/analyze
    separately on the chosen result for a rich AI critique.
    """
    logger.info(f"Generate request - user={user_id} occasion={body.occasion}")

    outfits = await run_stylist_pipeline(
        user_id=str(user_id),
        occasion=body.occasion.value,
        mood=body.mood.value if body.mood else None,
        latitude=body.latitude,
        longitude=body.longitude,
        excluded_item_ids=[str(i) for i in body.excluded_item_ids],
        max_results=body.max_results
    )

    return SuccessResponse(data={
        "outfits": outfits,
        "generated_at": datetime.now(UTC).isoformat(),
        "occasion": body.occasion.value,
        "mood": body.mood.value if body.mood else None,
        "count": len(outfits)
    })

@router.post("/dress-me-for-this", response_model=SuccessResponse)
async def dress_me_for_this(
    body: DressMeForThisRequest,
    user_id: UUID = Depends(get_current_user_id)
):
    """
    Text input -> GPT extracts occasion/mood -> same stylist pipeline
    as /generate. GPT never sees wardrobe data; it only translates
    the user's sentence into the same structured inputs the
    Recommendation Engine already understands.
    """
    logger.info(f"Dress-me-for-this request - user={user_id} text='{body.text[:100]}'")

    intent = extract_intent(body.text)

    outfits = await run_stylist_pipeline(
        user_id=str(user_id),
        occasion=intent.occasion.value,
        mood=intent.mood.value if intent.mood else None,
        latitude=body.latitude,
        longitude=body.longitude,
        excluded_item_ids=[str(i) for i in body.excluded_item_ids],
        max_results=body.max_results
    )

    return SuccessResponse(data={
        "outfits": outfits,
        "generated_at": datetime.now(UTC).isoformat(),
        "interpreted_occasion": intent.occasion.value,
        "interpreted_mood": intent.mood.value if intent.mood else None,
        "interpretation_confidence": intent.confidence,
        "count": len(outfits)
    })

@router.post("/feedback", response_model=SuccessResponse)
async def stylist_feedback(
    body: FeedbackRequest,
    response: Response,
    user_id: UUID = Depends(get_current_user_id)
):
    """
    Record what the user did with a suggestion (saved / skipped / worn /
    tried_on / swapped). Stored for ranking; generation does not use it yet.
    """
    event = record_feedback(
        user_id=user_id,
        action=body.action,
        item_ids=body.item_ids,
        occasion=body.occasion,
        mood=body.mood,
        source=body.source,
        replaced_item_id=body.replaced_item_id,
        replacement_item_id=body.replacement_item_id,
    )
    response.status_code = 201
    return SuccessResponse(data=event)