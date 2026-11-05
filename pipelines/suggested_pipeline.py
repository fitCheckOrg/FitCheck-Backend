"""
N2 Suggested creators to follow (2026-11-03).

GET /api/users/suggested?limit=20
  -> {items: [user + counts {followers, posts} + preview_posts [{post_id, items}] + reason]}

Used by the last onboarding step and the empty Following tab.

Only people with at least one public post (not under review). Never: me,
people I already follow, people blocked either way or hidden by me.

Score for each creator:
    +3  per style tag we share (profiles.style_tags, B10)
    +2  per post of theirs in a category I like (cap 3 posts) - categories I like
        come from my style tags (mapped below) and posts I've reacted to
    +1  per reaction their posts got in the last 7 days (cap 10)
    then most followers, then newest post.
No signals (new user) -> it is simply "popular creators".
"""

import logging

from core.storage.supabase_client import supabase
from pipelines.safety_pipeline import excluded_user_ids

logger = logging.getLogger(__name__)

POST_POOL = 300
CHUNK = 80
PREVIEW_POSTS = 3
PREVIEW_PIECES = 3
DEFAULT_LIMIT = 20
MAX_LIMIT = 50
STYLE_TO_CATEGORIES = {
    "casual": {"everyday", "weekend"},
    "formal": {"formal", "date"},
    "streetwear": {"street"},
    "business": {"work"},
    "athletic": {"sport"},
    "vintage": set(),
}


def _chunks(ids: list[str]):
    for i in range(0, len(ids), CHUNK):
        yield ids[i:i + CHUNK]


def _my_signals(me: str) -> tuple[set[str], set[str]]:
    """(my style tags, categories I like)."""
    prof = supabase.table("profiles").select("style_tags").eq("id", me).limit(1).execute().data
    tags = {str(t).lower() for t in ((prof[0].get("style_tags") if prof else None) or [])}
    cats = set()
    for t in tags:
        cats |= STYLE_TO_CATEGORIES.get(t, set())
    reacted = supabase.table("outfit_reactions").select("outfit_id")\
        .eq("user_id", me).order("created_at", desc=True).limit(50).execute().data or []
    ids = [str(r["outfit_id"]) for r in reacted]
    for part in _chunks(ids):
        for p in supabase.table("outfits").select("category").in_("outfit_id", part).execute().data or []:
            if p.get("category"):
                cats.add(p["category"])
    return tags, cats


def suggested_creators(user_id, limit: int = DEFAULT_LIMIT) -> dict:
    # Imported here to avoid circular imports.
    from pipelines.users_pipeline import build_users
    from api.social import _get_bulk_item_details

    me = str(user_id)
    limit = max(1, min(int(limit), MAX_LIMIT))

    following = {str(r["following_id"]) for r in supabase.table("follows").select("following_id")
                 .eq("follower_id", me).execute().data or []}
    skip = following | excluded_user_ids(me) | {me}

    posts = supabase.table("outfits")\
        .select("outfit_id, user_id, category, item_ids, published_at")\
        .eq("is_public", True).eq("under_review", False)\
        .order("published_at", desc=True, nullsfirst=False).limit(POST_POOL).execute().data or []
    by_creator: dict[str, list[dict]] = {}
    for p in posts:
        uid = str(p["user_id"])
        if uid not in skip:
            by_creator.setdefault(uid, []).append(p)
    if not by_creator:
        return {"items": []}
    creators = list(by_creator)

    my_tags, my_cats = _my_signals(me)

    their_tags: dict[str, set[str]] = {}
    followers: dict[str, int] = {c: 0 for c in creators}
    for part in _chunks(creators):
        for r in supabase.table("profiles").select("id, style_tags").in_("id", part).execute().data or []:
            their_tags[str(r["id"])] = {str(t).lower() for t in (r.get("style_tags") or [])}
        for f in supabase.table("follows").select("following_id").in_("following_id", part).execute().data or []:
            followers[str(f["following_id"])] = followers.get(str(f["following_id"]), 0) + 1

    recent: dict[str, int] = {}
    post_ids = [str(p["outfit_id"]) for c in creators for p in by_creator[c]]
    owner = {str(p["outfit_id"]): c for c in creators for p in by_creator[c]}
    for part in _chunks(post_ids):
        for r in supabase.table("public_post_rank").select("outfit_id, reactions_7d")\
                .in_("outfit_id", part).execute().data or []:
            c = owner.get(str(r["outfit_id"]))
            if c:
                recent[c] = recent.get(c, 0) + int(r.get("reactions_7d") or 0)

    def score(c: str) -> tuple:
        shared = my_tags & their_tags.get(c, set())
        liked = sum(1 for p in by_creator[c] if p.get("category") in my_cats)
        s = 3 * len(shared) + 2 * min(liked, 3) + min(recent.get(c, 0), 10)
        newest = str(by_creator[c][0].get("published_at") or "")
        return (s, followers.get(c, 0), newest)

    creators.sort(key=score, reverse=True)
    picked = creators[:limit]

    preview = {c: by_creator[c][:PREVIEW_POSTS] for c in picked}
    pieces = _get_bulk_item_details(list({i for ps in preview.values() for p in ps
                                          for i in (p.get("item_ids") or [])[:PREVIEW_PIECES]}))
    people = build_users(picked, me)

    items = []
    for c in picked:
        if c not in people:
            continue
        shared = sorted(my_tags & their_tags.get(c, set()))
        liked = [p.get("category") for p in by_creator[c] if p.get("category") in my_cats]
        if shared:
            reason = "Shares your style: " + ", ".join(shared)
        elif liked:
            reason = "Posts " + liked[0].replace("_", " ") + " looks"
        elif recent.get(c):
            reason = "Popular this week"
        else:
            reason = "New on FitCheck"
        items.append({
            **people[c],
            "counts": {"followers": followers.get(c, 0), "posts": len(by_creator[c])},
            "preview_posts": [{
                "post_id": str(p["outfit_id"]),
                "items": [pieces.get(i, {"item_id": i, "deleted": True})
                          for i in (p.get("item_ids") or [])[:PREVIEW_PIECES]],
            } for p in preview[c]],
            "reason": reason,
        })
    return {"items": items}