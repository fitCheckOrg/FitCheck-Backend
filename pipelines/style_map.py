"""
ONB-4 E (2026-11-14): which Discover post categories suit each style word and
each "love to see" pick. Shared by suggested creators and the For you feed.

Post categories (shared.models.enums.DiscoverCategory): editorial, streetwear,
minimal, formal, vintage, afrocentric, date_night, y2k_revival, festival.
A category that doesn't exist simply never matches.
"""

STYLE_TO_CATEGORIES = {
    "casual": {"minimal", "streetwear"},
    "minimalist": {"minimal"},
    "streetwear": {"streetwear"},
    "athleisure": {"streetwear"},
    "sporty": {"streetwear"},
    "smart_casual": {"minimal", "editorial"},
    "business": {"formal", "minimal"},
    "formal": {"formal"},
    "old_money": {"formal", "vintage"},
    "luxury": {"editorial", "formal"},
    "avant_garde": {"editorial"},
    "y2k": {"y2k_revival"},
    "vintage": {"vintage"},
    "contemporary": {"editorial", "minimal"},
    "traditional": {"afrocentric"},
    "gender_fluid": {"editorial"},
}

# setup.discover picks that point at a kind of post (the rest are about people / tips).
DISCOVER_TO_CATEGORIES = {
    "everyday": {"minimal", "streetwear"},
    "photography": {"editorial"},
    "new_collections": {"editorial"},
}


def categories_for(style_tags, discover=()) -> set[str]:
    cats = set()
    for t in style_tags or ():
        cats |= STYLE_TO_CATEGORIES.get(t, set())
    for d in discover or ():
        cats |= DISCOVER_TO_CATEGORIES.get(d, set())
    return cats