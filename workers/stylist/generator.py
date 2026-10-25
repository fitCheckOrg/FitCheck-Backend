import random
import hashlib
from datetime import datetime, UTC
from shared.logger import get_logger
from workers.outfit.compatibility import calculate_compatibility
from workers.outfit.validator import validate_outfit_composition
from workers.stylist.outfit_key import outfit_key
from shared.exceptions import FitCheckException
from core.config.settings import settings

logger = get_logger(__name__)

MOOD_STYLE_PRIORITY = {
    "confident": ["streetwear", "formal"],
    "minimal": ["casual", "business"],
    "relaxed": ["casual"],
    "professional": ["business", "formal"],
}

MAX_ITEMS_PER_CATEGORY = 6

# The Stylist accepts more occasions (OutfitOccasion) than the wardrobe AI
# tags items with (everyday|work|party|sport|formal|outdoor). Without this
# map, "date" and "travel" matched no item at all -> NO_VALID_COMBINATION.
OCCASION_EQUIVALENTS = {
    "date": {"date", "party", "formal", "everyday"},
    "travel": {"travel", "everyday", "outdoor"},
}

MAX_SWAP_RESULTS = 5
SWAP_MOOD_BONUS = 3  # rank-only nudge for swap candidates matching the mood

# Style profile (DNA-3). RANK-ONLY nudges, never added to the score shown
# to the user - same rule as the rotation signal.
STYLE_MATCH_BONUS = 2        # per item whose style is one the user picked
PREFERRED_COLOR_BONUS = 1.5  # per item in a colour the user likes
DISLIKED_COLOR_PENALTY = 6   # per item in a colour the user dislikes


def _suits_occasion(item: dict, occasion: str) -> bool:
    if occasion == "everyday":
        return True
    accepted = OCCASION_EQUIVALENTS.get(occasion, {occasion})
    return bool(accepted & set(item.get("occasions") or []))


def generate_outfits(
    wardrobe_items: list[dict],
    occasion: str = "everyday",
    mood: str | None = None,
    weather: dict | None = None,
    excluded_item_ids: list[str] | None = None,
    max_results: int = 3,
    user_id: str | None = None,
    variation: int = 0,
    excluded_outfit_keys: list[str] | None = None,
    style_profile: dict | None = None,
) -> list[dict]:
    """
    Same request + same day + same variation -> same results (deterministic).
    style_profile (styles / preferred_colors / disliked_colors) nudges RANKING
    only; the shown score stays the pure compatibility score.
    variation > 0 reshuffles (still only top-scored outfits).
    excluded_outfit_keys drops exact combinations the user has already seen.
    """
    excluded = set(excluded_item_ids or [])
    excluded_keys = {k.lower() for k in (excluded_outfit_keys or [])}
    logger.info(
        "Generating outfits - occasion=%s mood=%s items_available=%d variation=%d excluded_keys=%d",
        occasion, mood, len(wardrobe_items), variation, len(excluded_keys)
    )

    # ONE seeded RNG shared across the whole pipeline.
    # variation=0 keeps the original per-user-per-day seed exactly.
    seed_key = f"{user_id or 'anon'}-{datetime.now(UTC).date().isoformat()}"
    if variation:
        seed_key = f"{seed_key}-v{variation}"
    seed = int(hashlib.sha256(seed_key.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)

    pool = [
        item for item in wardrobe_items
        if item["item_id"] not in excluded
        and _suits_occasion(item, occasion)
    ]

    if weather:
        pool = _filter_by_weather(pool, weather)

    if mood:
        pool = _prioritize_by_mood(pool, mood)

    by_category = _group_by_category(pool)
    by_category = _prune_categories(by_category, rng, style_profile)
    candidates = _build_candidates(by_category)

    scored = []
    skipped_seen = 0
    for candidate_items in candidates:
        try:
            validate_outfit_composition(candidate_items)
        except FitCheckException:
            continue

        key = outfit_key(item["item_id"] for item in candidate_items)
        if key in excluded_keys:
            skipped_seen += 1
            continue

        scores = calculate_compatibility(candidate_items)
        compatibility_score = scores["overall_score"]
        rotation_score = _calculate_rotation_score(candidate_items)
        preference_score, personal_notes = _style_preference(candidate_items, style_profile)
        final_rank_score = compatibility_score + rotation_score + preference_score

        scored.append({
            "item_ids": [item["item_id"] for item in candidate_items],
            "outfit_key": key,
            "items": candidate_items,
            "score": compatibility_score,
            "grade": scores["overall_grade"],
            "strengths": scores["strengths"],
            "warnings": scores["warnings"],
            "reason": _build_reason(scores, mood),
            "personal_notes": personal_notes,
            "_compatibility_score": compatibility_score,
            "_rotation_score": rotation_score,
            "_final_rank_score": final_rank_score,
        })

    if not scored:
        if skipped_seen:
            logger.warning("All %d valid combinations were already seen", skipped_seen)
            raise FitCheckException(
                "You've seen every outfit we can make for this right now. "
                "Try another occasion or mood, or add more items.",
                code="NO_NEW_COMBINATIONS", status_code=400
            )
        logger.warning("No valid outfit candidates survived scoring")
        raise FitCheckException(
            "We couldn't put together an outfit for this occasion right now. "
            "Try a different occasion or mood.",
            code="NO_VALID_COMBINATION", status_code=400
        )

    top_results = _select_from_elite_cluster(scored, max_results, rng)

    for r in top_results:
        r.pop("_compatibility_score", None)
        r.pop("_rotation_score", None)
        r.pop("_final_rank_score", None)

    logger.info(
        "Generated %d candidates, returning top %d - best score=%s",
        len(scored), len(top_results), top_results[0]["score"]
    )

    return top_results


def suggest_swaps(
    outfit_items: list[dict],
    replace_item: dict,
    wardrobe_items: list[dict],
    occasion: str = "everyday",
    mood: str | None = None,
    excluded_item_ids: list[str] | None = None,
    max_results: int = 3,
    style_profile: dict | None = None,
) -> dict:
    """
    Replace ONE item in an outfit with the best same-category alternatives
    from the (already active-only) wardrobe. Each alternative is the full
    re-scored outfit, plus how much it changes the score.
    """
    max_results = max(1, min(int(max_results), MAX_SWAP_RESULTS))
    outfit_ids = {i["item_id"] for i in outfit_items}
    excluded = set(excluded_item_ids or [])
    category = replace_item.get("category")
    kept = [i for i in outfit_items if i["item_id"] != replace_item["item_id"]]

    current_score = None
    if len(outfit_items) >= 2:
        current_score = calculate_compatibility(outfit_items)["overall_score"]

    same_category = [
        i for i in wardrobe_items
        if i.get("category") == category
        and i["item_id"] not in outfit_ids
        and i["item_id"] not in excluded
    ]
    # Prefer items suited to the occasion; fall back to all of the category.
    for_occasion = [i for i in same_category if _suits_occasion(i, occasion)]
    pool = for_occasion or same_category
    mood_styles = set(MOOD_STYLE_PRIORITY.get((mood or "").lower(), []))

    ranked = []
    for candidate in pool:
        new_items = kept + [candidate]
        try:
            validate_outfit_composition(new_items)
        except FitCheckException:
            continue

        scores = calculate_compatibility(new_items)
        preference_score, personal_notes = _style_preference([candidate], style_profile)
        rank = scores["overall_score"] + _calculate_rotation_score([candidate]) + preference_score
        if candidate.get("style") in mood_styles:
            rank += SWAP_MOOD_BONUS

        new_ids = [i["item_id"] for i in new_items]
        ranked.append((rank, {
            "item_id": candidate["item_id"],
            "item": candidate,
            "item_ids": new_ids,
            "outfit_key": outfit_key(new_ids),
            "score": scores["overall_score"],
            "grade": scores["overall_grade"],
            "score_change": (scores["overall_score"] - current_score) if current_score is not None else None,
            "reason": _build_swap_reason(scores, current_score, mood),
            "personal_notes": personal_notes,
        }))

    ranked.sort(key=lambda r: r[0], reverse=True)
    alternatives = [r[1] for r in ranked[:max_results]]

    return {
        "replaced_item_id": replace_item["item_id"],
        "category": category,
        "current_score": current_score,
        "alternatives": alternatives,
        "count": len(alternatives),
        "message": None if alternatives else
            f"No other {category or ''} item in your active wardrobe fits this outfit.".replace("  ", " "),
    }


def _prune_categories(
    by_category: dict[str, list[dict]],
    rng: random.Random,
    style_profile: dict | None = None,
) -> dict[str, list[dict]]:
    """
    Keep at most MAX_ITEMS_PER_CATEGORY per category: favourites first, then
    items the style profile prefers (so a liked colour isn't dropped before
    scoring even starts), then least worn. No profile -> original ordering.
    """
    pruned = {}
    for cat, items in by_category.items():
        shuffled = items[:]
        rng.shuffle(shuffled)   # seeded rng, not global random
        sorted_items = sorted(
            shuffled,
            key=lambda i: (
                not i.get("favorite", False),
                -_style_preference([i], style_profile)[0],
                i.get("times_worn", 0)
            )
        )
        pruned[cat] = sorted_items[:MAX_ITEMS_PER_CATEGORY]
    return pruned


def _select_from_elite_cluster(
    scored: list[dict],
    max_results: int,
    rng: random.Random
) -> list[dict]:
    scored.sort(key=lambda x: x["_final_rank_score"], reverse=True)

    if not scored:
        return []

    top_score = scored[0]["_final_rank_score"]
    band = settings.ROTATION_SCORE_BAND

    elite_cluster = [c for c in scored if top_score - c["_final_rank_score"] <= band]
    remainder = [c for c in scored if c not in elite_cluster]

    rng.shuffle(elite_cluster)   # same rng instance as _prune_categories used

    ordered = elite_cluster + remainder
    return ordered[:max_results]

def _calculate_rotation_score(items: list[dict]) -> float:
    """
    A SEPARATE signal from compatibility - never mixed into the
    score shown to the user. Uses only confirmed-real fields
    (favorite, times_worn, last_worn), no new backend data required.
    """
    adjustment = 0.0

    for item in items:
        if item.get("favorite"):
            adjustment += settings.FAVORITE_BONUS

        times_worn = item.get("times_worn") or 0
        adjustment -= min(times_worn * settings.TIMES_WORN_WEIGHT, 10)

        last_worn = item.get("last_worn")
        if last_worn:
            try:
                worn_dt = datetime.fromisoformat(last_worn.replace("Z", "+00:00"))
                days_since = (datetime.now(UTC) - worn_dt).days
                if days_since < settings.RECENTLY_WORN_DAYS:
                    adjustment -= settings.ROTATION_PENALTY
            except (ValueError, TypeError):
                pass

    return adjustment


def _norm_list(values) -> list[str]:
    return [str(v).strip().lower() for v in (values or []) if str(v).strip()]


# Colour families. The quiz sends one word per family (black, white, grey,
# navy, blue, beige, brown, green, olive, red, burgundy, pink, purple,
# yellow, orange); the wardrobe AI writes free-form names ("gray",
# "light gray", "maroon", "camouflage"...). Both sides are mapped to a
# family before comparing. Unknown names only match themselves.
COLOR_FAMILIES = {
    "black": ["black", "jet", "onyx"],
    "white": ["white", "snow"],
    "grey": ["grey", "charcoal", "silver", "slate", "ash", "heather"],
    "navy": ["navy", "navy blue", "dark blue", "midnight"],
    "blue": ["blue", "denim", "sky", "cobalt", "azure", "teal"],
    "beige": ["beige", "cream", "khaki", "sand", "camel", "ivory", "nude",
              "off white", "taupe", "stone"],
    "brown": ["brown", "tan", "chocolate", "coffee", "mocha", "cognac"],
    "green": ["green", "emerald", "mint", "sage", "forest", "lime"],
    "olive": ["olive", "camouflage", "camo", "army", "khaki green", "army green"],
    "red": ["red", "crimson", "scarlet"],
    "burgundy": ["burgundy", "maroon", "wine", "oxblood"],
    "pink": ["pink", "rose", "blush", "fuchsia", "magenta"],
    "purple": ["purple", "violet", "lavender", "lilac", "plum", "mauve"],
    "yellow": ["yellow", "mustard", "gold", "lemon"],
    "orange": ["orange", "rust", "coral", "peach", "terracotta"],
}
_COLOR_LOOKUP = {alias: family for family, aliases in COLOR_FAMILIES.items() for alias in aliases}


def color_family(name) -> str | None:
    """'light gray' -> 'grey', 'navy blue' -> 'navy', 'maroon' -> 'burgundy'."""
    n = str(name or "").strip().lower().replace("-", " ").replace("gray", "grey")
    n = " ".join(n.split())
    if not n:
        return None
    if n in _COLOR_LOOKUP:
        return _COLOR_LOOKUP[n]
    for word in n.split():          # first recognised word wins: "olive green" -> olive
        if word in _COLOR_LOOKUP:
            return _COLOR_LOOKUP[word]
    return n


def _color_matches(item: dict, colors: list[str]) -> str | None:
    """Return the profile colour (family) this item matches, if any."""
    item_family = color_family(item.get("dominant_color"))
    if not item_family:
        return None
    for c in colors:
        if color_family(c) == item_family:
            return c
    return None


def _style_preference(items: list[dict], style_profile: dict | None) -> tuple[float, list[str]]:
    """
    Rank adjustment + short user-facing notes from the style profile.
    No profile / empty profile -> (0, []), i.e. exactly the old behaviour.
    """
    if not style_profile:
        return 0.0, []

    styles = set(_norm_list(style_profile.get("styles")))
    liked = _norm_list(style_profile.get("preferred_colors"))
    disliked = _norm_list(style_profile.get("disliked_colors"))
    if not (styles or liked or disliked):
        return 0.0, []

    adjustment = 0.0
    matched_styles: list[str] = []
    matched_colors: list[str] = []

    for item in items:
        style = str(item.get("style") or "").strip().lower()
        if style and style in styles:
            adjustment += STYLE_MATCH_BONUS
            if style not in matched_styles:
                matched_styles.append(style)

        bad = _color_matches(item, disliked)
        if bad:
            adjustment -= DISLIKED_COLOR_PENALTY
            continue
        good = _color_matches(item, liked)
        if good:
            adjustment += PREFERRED_COLOR_BONUS
            if good not in matched_colors:
                matched_colors.append(good)

    notes = []
    if matched_styles:
        notes.append(f"Matches your {' & '.join(matched_styles)} style")
    if matched_colors:
        notes.append(f"Uses {', '.join(matched_colors)}, a colour you like" if len(matched_colors) == 1
                     else f"Uses {', '.join(matched_colors)}, colours you like")
    return adjustment, notes


def _group_by_category(items: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for item in items:
        cat = item.get("category")
        grouped.setdefault(cat, []).append(item)
    return grouped


def _build_candidates(by_category: dict[str, list[dict]]) -> list[list[dict]]:
    from itertools import product
    candidates = []
    shoes = by_category.get("shoes") or [None]

    if by_category.get("dress"):
        for dress, shoe in product(by_category["dress"], shoes):
            combo = [dress] + ([shoe] if shoe else [])
            candidates.append(combo)

    if by_category.get("top") and by_category.get("bottom"):
        for top, bottom, shoe in product(
            by_category["top"], by_category["bottom"], shoes
        ):
            combo = [top, bottom] + ([shoe] if shoe else [])
            candidates.append(combo)

    return candidates


def _filter_by_weather(items: list[dict], weather: dict) -> list[dict]:
    temp = weather.get("temp_celsius")
    if temp is None:
        return items
    if temp >= 25:
        target_seasons = {"summer", "spring"}
    elif temp <= 12:
        target_seasons = {"winter", "fall"}
    else:
        target_seasons = {"spring", "fall", "summer", "winter"}
    filtered = [
        item for item in items
        if not item.get("seasons") or set(item["seasons"]) & target_seasons
    ]
    return filtered if filtered else items


def _prioritize_by_mood(items: list[dict], mood: str) -> list[dict]:
    preferred_styles = MOOD_STYLE_PRIORITY.get(mood.lower())
    if not preferred_styles:
        return items
    preferred = [item for item in items if item.get("style") in preferred_styles]
    others = [item for item in items if item.get("style") not in preferred_styles]
    return preferred + others


def _build_reason(scores: dict, mood: str | None) -> str:
    parts = []
    if scores["color_score"] >= 85:
        parts.append("strong color harmony")
    if scores["style_score"] >= 85:
        parts.append("consistent style")
    if scores["occasion_score"] >= 85:
        parts.append("great fit for the occasion")
    if mood:
        parts.append(f"suits a {mood} mood")
    if not parts:
        return "A balanced option from your wardrobe."
    return "Matches your wardrobe with " + ", ".join(parts) + "."


def _build_swap_reason(scores: dict, current_score: int | None, mood: str | None) -> str:
    if current_score is not None:
        change = scores["overall_score"] - current_score
        if change > 0:
            lead = f"Raises the outfit score by {change}"
        elif change < 0:
            lead = f"Changes the look; score drops by {-change}"
        else:
            lead = "Keeps the same score with a different look"
    else:
        lead = "A good fit for this outfit"
    detail = scores["strengths"][0].rstrip(".").lower() if scores["strengths"] else None
    return f"{lead} - {detail}." if detail else f"{lead}."