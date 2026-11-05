"""
D16 Related posts - "More like this" (2026-11-03).

GET /api/social/posts/{post_id}/related?limit=12 -> {items: [feed item]}

Ranking, for each candidate public post:
    +3   same category as the source post
    +1   per shared piece colour   (max +3)
    +1   per shared piece type     (max +3)
    then reactions in the last 7 days, then newest.

Candidates: the newest CANDIDATES public posts that are not under review,
not the source post, not mine, not from anyone blocked either way / hidden,
and not marked "not interested". Same feed-item shape as /social/feed.
"""

import logging

from core.storage.supabase_client import supabase
from pipelines.safety_pipeline import (
    _visible_post, excluded_user_ids, not_interested_post_ids, not_in,
)

logger = logging.getLogger(__name__)

CANDIDATES = 150
CHUNK = 80              # ids per IN (...) query, keeps URLs short
SAME_CATEGORY = 3
MAX_SHARED = 3
DEFAULT_LIMIT = 12
MAX_LIMIT = 30


def _chunks(ids: list[str]):
    for i in range(0, len(ids), CHUNK):
        yield ids[i:i + CHUNK]


def _piece_tags(item_ids: list[str]) -> dict[str, tuple[str | None, str | None]]:
    """item_id -> (colour, type), lowercased."""
    out = {}
    ids = list({str(i) for i in item_ids if i})
    for part in _chunks(ids):
        rows = supabase.table("closet_items").select("item_id, dominant_color, item_type")\
            .in_("item_id", part).execute().data or []
        for r in rows:
            out[str(r["item_id"])] = (
                (r.get("dominant_color") or "").strip().lower() or None,
                (r.get("item_type") or "").strip().lower() or None,
            )
    return out


def _recent_reactions(post_ids: list[str]) -> dict[str, int]:
    out = {}
    for part in _chunks(post_ids):
        rows = supabase.table("public_post_rank").select("outfit_id, reactions_7d")\
            .in_("outfit_id", part).execute().data or []
        for r in rows:
            out[str(r["outfit_id"])] = int(r.get("reactions_7d") or 0)
    return out


def related_posts(post_id, viewer_id, limit: int = DEFAULT_LIMIT) -> dict:
    # Imported here to avoid a circular import.
    from api.social import FEED_FIELDS, _decorate_posts

    pid, me = str(post_id), str(viewer_id)
    _visible_post(pid, me)                                   # 404 POST_NOT_FOUND
    limit = max(1, min(int(limit), MAX_LIMIT))

    src = supabase.table("outfits").select("outfit_id, category, item_ids")\
        .eq("outfit_id", pid).limit(1).execute().data[0]

    q = supabase.table("outfits")\
        .select("outfit_id, user_id, category, item_ids, published_at")\
        .eq("is_public", True).eq("under_review", False)\
        .neq("outfit_id", pid).neq("user_id", me)
    q = not_in(q, "user_id", excluded_user_ids(me))
    q = not_in(q, "outfit_id", not_interested_post_ids(me))
    candidates = q.order("published_at", desc=True, nullsfirst=False)\
        .limit(CANDIDATES).execute().data or []
    if not candidates:
        return {"items": []}

    all_items = list(src.get("item_ids") or []) + [i for c in candidates for i in (c.get("item_ids") or [])]
    tags = _piece_tags(all_items)

    def tag_sets(item_ids):
        colours = {tags[str(i)][0] for i in (item_ids or []) if str(i) in tags and tags[str(i)][0]}
        types = {tags[str(i)][1] for i in (item_ids or []) if str(i) in tags and tags[str(i)][1]}
        return colours, types

    src_colours, src_types = tag_sets(src.get("item_ids"))
    recent = _recent_reactions([str(c["outfit_id"]) for c in candidates])

    def score(c: dict) -> tuple:
        colours, types = tag_sets(c.get("item_ids"))
        s = (SAME_CATEGORY if src.get("category") and c.get("category") == src.get("category") else 0)
        s += min(len(colours & src_colours), MAX_SHARED) + min(len(types & src_types), MAX_SHARED)
        return (s, recent.get(str(c["outfit_id"]), 0), str(c.get("published_at") or ""))

    candidates.sort(key=score, reverse=True)
    picked = [str(c["outfit_id"]) for c in candidates[:limit]]

    rows = supabase.table("outfits").select(FEED_FIELDS).in_("outfit_id", picked).execute().data or []
    by_id = {str(r["outfit_id"]): r for r in rows}
    items = [by_id[i] for i in picked if i in by_id]       # keep the ranked order
    _decorate_posts(items, me)
    return {"items": items}