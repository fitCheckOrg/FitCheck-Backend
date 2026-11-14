"""
ONB-4 E (2026-11-14): which Discover post categories suit each style word and
each "love to see" pick. Shared by suggested creators and the For you feed.

Post categories (shared.models.enums.DiscoverCategory, A2 2026-10-31):
everyday, work, night_out, date, formal, street, travel, sport, weekend.
A category that doesn't exist simply never matches.
"""

STYLE_TO_CATEGORIES = {
    "casual": {"everyday", "weekend"},
    "minimalist": {"everyday", "work"},
    "streetwear": {"street"},
    "athleisure": {"sport", "everyday"},
    "sporty": {"sport"},
    "smart_casual": {"work", "date"},
    "business": {"work"},
    "formal": {"formal"},
    "old_money": {"formal", "date"},
    "luxury": {"formal", "night_out"},
    "avant_garde": {"night_out"},
    "y2k": {"street", "night_out"},
    "vintage": {"weekend"},
    "contemporary": {"everyday", "work"},
    "traditional": {"formal", "weekend"},
    "gender_fluid": {"street"},
}

# setup.discover picks that point at a kind of post (the rest are about people / tips).
DISCOVER_TO_CATEGORIES = {
    "everyday": {"everyday"},
}


def categories_for(style_tags, discover=()) -> set[str]:
    cats = set()
    for t in style_tags or ():
        cats |= STYLE_TO_CATEGORIES.get(t, set())
    for d in discover or ():
        cats |= DISCOVER_TO_CATEGORIES.get(d, set())
    return cats