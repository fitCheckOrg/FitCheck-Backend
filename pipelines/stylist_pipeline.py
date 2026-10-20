from shared.logger import get_logger
from shared.exceptions import FitCheckException, ClothingItemNotFoundError
from core.storage.supabase_client import supabase
from core.weather.client import weather_client
from workers.stylist.generator import generate_outfits, suggest_swaps
from workers.stylist.wardrobe_validator import analyze_wardrobe
from core.constants.wardrobe_select import WARDROBE_ITEM_SELECT

logger = get_logger(__name__)

MAX_OUTFIT_ITEMS = 10

# The style profile matches on colour names, so make sure dominant_color is loaded.
STYLIST_ITEM_SELECT = (
    WARDROBE_ITEM_SELECT if "dominant_color" in WARDROBE_ITEM_SELECT
    else f"{WARDROBE_ITEM_SELECT}, dominant_color"
)


def _active_wardrobe(user_id: str) -> list[dict]:
    result = supabase.table("closet_items")\
        .select(STYLIST_ITEM_SELECT)\
        .eq("user_id", user_id)\
        .eq("is_archived", False)\
        .execute()
    return result.data or []


def _style_profile(user_id: str) -> dict:
    """
    The user's style profile (DNA-3), or {} if they haven't onboarded.
    Never fails generation: any error just means "no personalisation".
    """
    try:
        res = supabase.table("style_profiles")\
            .select("styles, preferred_colors, disliked_colors")\
            .eq("user_id", user_id)\
            .maybe_single()\
            .execute()
        return (res.data or {}) if res else {}
    except Exception as exc:
        logger.warning("Style profile unavailable for user=%s: %s", user_id, exc)
        return {}


async def run_stylist_pipeline(
    user_id: str,
    occasion: str = "everyday",
    mood: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    excluded_item_ids: list[str] | None = None,
    max_results: int = 3,
    variation: int = 0,
    excluded_outfit_keys: list[str] | None = None,
) -> list[dict]:
    """
    Orchestrates the full recommendation flow:
    Weather -> Wardrobe fetch -> Wardrobe validation ->
    Generator -> Compatibility scoring -> Top results.
    """
    logger.info(
        "Stylist pipeline started - user=%s occasion=%s mood=%s variation=%s",
        user_id, occasion, mood, variation
    )

    weather = await weather_client.get_weather(latitude, longitude)

    wardrobe_items = _active_wardrobe(user_id)
    style_profile = _style_profile(user_id)

    analysis = analyze_wardrobe(wardrobe_items)
    if not analysis["is_viable"]:
        missing = analysis["missing_categories"]
        message = (
            f"Add {' and '.join(missing)} to your wardrobe to unlock "
            f"outfit suggestions." if missing
            else "Add a few more items to unlock outfit suggestions."
        )
        logger.warning("Wardrobe not viable - missing=%s", missing)
        raise FitCheckException(
            message,
            code="INSUFFICIENT_WARDROBE",
            status_code=400
        )

    results = generate_outfits(
        wardrobe_items=wardrobe_items,
        occasion=occasion,
        mood=mood,
        weather=weather,
        excluded_item_ids=excluded_item_ids,
        max_results=max_results,
        user_id=user_id,
        variation=variation,
        excluded_outfit_keys=excluded_outfit_keys,
        style_profile=style_profile,
    )

    logger.info(
        "Stylist pipeline complete - user=%s outfits=%d weather_used=%s personalised=%s",
        user_id, len(results), weather is not None, bool(style_profile)
    )

    return results


def run_swap_pipeline(
    user_id: str,
    item_ids: list[str],
    replace_item_id: str,
    occasion: str = "everyday",
    mood: str | None = None,
    excluded_item_ids: list[str] | None = None,
    max_results: int = 3,
) -> dict:
    """Replace ONE item of an outfit with alternatives from the active wardrobe."""
    ids = list(dict.fromkeys(item_ids))
    if len(ids) < 2:
        raise FitCheckException(
            "Swapping needs an outfit of at least 2 items.",
            code="SWAP_NEEDS_OUTFIT", status_code=400
        )
    if len(ids) > MAX_OUTFIT_ITEMS:
        raise FitCheckException(
            f"An outfit can contain at most {MAX_OUTFIT_ITEMS} items.",
            code="TOO_MANY_ITEMS", status_code=400
        )
    if replace_item_id not in ids:
        raise FitCheckException(
            "replace_item_id must be one of item_ids.",
            code="SWAP_ITEM_NOT_IN_OUTFIT", status_code=400
        )

    # The outfit's own items may include archived ones; ownership is what matters.
    outfit_result = supabase.table("closet_items")\
        .select(STYLIST_ITEM_SELECT)\
        .eq("user_id", user_id)\
        .in_("item_id", ids)\
        .execute()
    outfit_items = outfit_result.data or []
    if len(outfit_items) != len(ids):
        raise ClothingItemNotFoundError()

    by_id = {i["item_id"]: i for i in outfit_items}
    outfit_items = [by_id[i] for i in ids]  # keep request order

    logger.info("Swap request - user=%s replace=%s items=%d", user_id, replace_item_id, len(ids))

    return suggest_swaps(
        outfit_items=outfit_items,
        replace_item=by_id[replace_item_id],
        wardrobe_items=_active_wardrobe(user_id),
        occasion=occasion,
        mood=mood,
        excluded_item_ids=excluded_item_ids,
        max_results=max_results,
        style_profile=_style_profile(user_id),
    )