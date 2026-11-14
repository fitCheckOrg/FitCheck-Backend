"""
ONB-4 E: Discover "For you" ranked by the user's picks (2026-11-14).

GET /api/social/feed/for-you?limit=&offset=&category=
  -> {outfits, pagination, tuned_to?, thin?}      (same post shape as /social/feed)

How a page is built:
    1. Pool = the newest POOL public posts (not under review, nobody blocked /
       hidden, no posts I marked "not interested"), optional category.
    2. My signals = my style words (profiles.style_tags) + my "love to see" picks
       (profiles.setup_discover) -> the Discover categories that suit them
       (pipelines/style_map.py).
    3. Each pool post scores:
           +3 per style word its creator shares with me
           +2 if its category suits one of my style words
           +1 if its category suits one of my "love to see" picks
       Posts that score > 0 come first (score, then reactions this week, then
       newest). Then the rest, most popular first (all-time reactions, then newest).
    4. No signals at all -> plain newest first (same as sort=recent), no tuned_to.
    5. Past the pool, pages continue with older posts, newest first.

First page only (offset 0, no category):
    tuned_to  my style word ids that matched posts, most matches first (all my
              words if none matched). Omitted when no personal ranking was applied.
    thin      true when fewer than THIN_BELOW posts match, so popular posts fill
              the rest.
PRIVACY: tuned_to only echoes the caller's own style words, to the caller.
"""

import logging

from core.storage.supabase_client import supabase
from pipelines.safety_pipeline import excluded_user_ids, not_interested_post_ids, not_in
from pipelines.style_map import STYLE_TO_CATEGORIES, DISCOVER_TO_CATEGORIES, categories_for

logger = logging.getLogger(__name__)

POOL = 500
CHUNK = 80
THIN_BELOW = 8


def _chunks(ids: list[str]):
    for i in range(0, len(ids), CHUNK):
        yield ids[i:i + CHUNK]


def _base_query(fields: str, category, excluded, skipped, count: bool = False):
    q = supabase.table("outfits").select(fields, count="exact") if count \
        else supabase.table("outfits").select(fields)
    q = q.eq("is_public", True).eq("under_review", False)
    if category:
        q = q.eq("category", category)
    q = not_in(q, "user_id", excluded)
    return not_in(q, "outfit_id", skipped)


def _my_signals(me: str) -> tuple[list[str], list[str]]:
    from pipelines.users_pipeline import normalize_style_tags
    rows = supabase.table("profiles").select("style_tags, setup_discover")\
        .eq("id", me).limit(1).execute().data
    row = rows[0] if rows else {}
    return normalize_style_tags(row.get("style_tags")), list(row.get("setup_discover") or [])


def for_you_feed(user_id, limit: int = 20, offset: int = 0, category: str | None = None) -> dict:
    from api.social import FEED_FIELDS, _decorate_posts
    from pipelines.users_pipeline import normalize_style_tags

    me = str(user_id)
    excluded = excluded_user_ids(me)
    skipped = not_interested_post_ids(me)

    res = _base_query("outfit_id, user_id, category, published_at", category, excluded, skipped, count=True)\
        .order("published_at", desc=True, nullsfirst=False).limit(POOL).execute()
    pool = res.data or []
    total = res.count or 0

    my_tags, my_discover = _my_signals(me)
    style_cats = categories_for(my_tags)
    discover_cats = {c for d in my_discover for c in DISCOVER_TO_CATEGORIES.get(d, set())}
    personal = bool(my_tags or discover_cats)

    matched_ids: list[str] = []
    contributions = {t: 0 for t in my_tags}
    if personal and pool:
        ids = [str(p["outfit_id"]) for p in pool]
        creators = list({str(p["user_id"]) for p in pool})
        their_tags: dict[str, set[str]] = {}
        for part in _chunks(creators):
            for r in supabase.table("profiles").select("id, style_tags").in_("id", part).execute().data or []:
                their_tags[str(r["id"])] = set(normalize_style_tags(r.get("style_tags")))
        rank: dict[str, dict] = {}
        for part in _chunks(ids):
            for r in supabase.table("public_post_rank").select("outfit_id, total_reactions, reactions_7d")\
                    .in_("outfit_id", part).execute().data or []:
                rank[str(r["outfit_id"])] = r

        scored, rest = [], []
        for p in pool:
            pid, cat = str(p["outfit_id"]), p.get("category")
            shared = set(my_tags) & their_tags.get(str(p["user_id"]), set())
            s = 3 * len(shared) + (2 if cat in style_cats else 0) + (1 if cat in discover_cats else 0)
            r = rank.get(pid, {})
            when = str(p.get("published_at") or "")
            if s > 0:
                scored.append(((s, int(r.get("reactions_7d") or 0), when), pid))
                for t in my_tags:
                    if t in shared or cat in STYLE_TO_CATEGORIES.get(t, set()):
                        contributions[t] += 1
            else:
                rest.append(((int(r.get("total_reactions") or 0), when), pid))
        scored.sort(key=lambda x: x[0], reverse=True)
        rest.sort(key=lambda x: x[0], reverse=True)
        matched_ids = [pid for _, pid in scored]
        ordered = matched_ids + [pid for _, pid in rest]
    else:
        ordered = [str(p["outfit_id"]) for p in pool]           # newest first

    # This page: from the ranked pool, then older posts past the pool (newest first).
    page_ids = ordered[offset:offset + limit]
    outfits = []
    if page_ids:
        rows = supabase.table("outfits").select(FEED_FIELDS).in_("outfit_id", page_ids).execute().data or []
        by_id = {str(r["outfit_id"]): r for r in rows}
        outfits = [by_id[i] for i in page_ids if i in by_id]
    need = limit - len(page_ids)
    if need > 0 and total > len(pool):
        start = len(pool) + max(0, offset - len(pool))
        outfits += _base_query(FEED_FIELDS, category, excluded, skipped)\
            .order("published_at", desc=True, nullsfirst=False)\
            .range(start, start + need - 1).execute().data or []

    _decorate_posts(outfits, me)
    data = {
        "outfits": outfits,
        "pagination": {"total": total, "limit": limit, "offset": offset,
                       "has_more": (offset + limit) < total},
    }
    if offset == 0 and not category and personal:
        hits = [t for t in my_tags if contributions.get(t)]
        hits.sort(key=lambda t: -contributions[t])               # stable: my order breaks ties
        data["tuned_to"] = hits or list(my_tags)
        data["thin"] = len(matched_ids) < THIN_BELOW
    logger.info("for you user=%s offset=%s personal=%s matched=%s", me, offset, personal, len(matched_ids))
    return data