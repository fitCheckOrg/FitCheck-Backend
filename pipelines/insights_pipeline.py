"""
E19 Insights (2026-11-03) - owner-only numbers for a post.

POST /api/social/posts/views          {"post_ids": [...]}       -> {ok, counted}
POST /api/social/posts/{id}/share     {"channel": image|system|link} -> {share_count}
GET  /api/social/posts/{id}/insights?range=7d|30d               -> numbers below

Tables (see the E19 SQL): post_views (post_id, viewer_id, day), post_shares,
and post_recreates (D17).

Views: each viewer counts at most once per post per day; the owner never
counts; only public posts that aren't under review; people blocked either
way are ignored. Saves = bookmarks + adds to a collection by other people.
Reactions in a range = reactions made (or last changed) in that range.
"""

import logging
from datetime import datetime, timedelta, timezone

from core.storage.supabase_client import supabase
from shared.exceptions import FitCheckException
from pipelines.safety_pipeline import _visible_post, excluded_user_ids

logger = logging.getLogger(__name__)

CHANNELS = ("image", "system", "link")
RANGES = {"7d": 7, "30d": 30}
MAX_VIEW_BATCH = 100
CHUNK = 80
REACTION_TYPES = ("love", "fire", "clean", "bold", "wear")


def _bad(code: str, message: str, status: int = 400) -> FitCheckException:
    return FitCheckException(message, code=code, status_code=status)


def _chunks(ids: list[str]):
    for i in range(0, len(ids), CHUNK):
        yield ids[i:i + CHUNK]


# ---------------------------------------------------------------- views

def record_views(post_ids: list, viewer_id) -> dict:
    me = str(viewer_id)
    ids = list(dict.fromkeys(str(p) for p in (post_ids or []) if p))[:MAX_VIEW_BATCH]
    if not ids:
        return {"ok": True, "counted": 0}
    excluded = excluded_user_ids(me)
    eligible = []
    for part in _chunks(ids):
        rows = supabase.table("outfits").select("outfit_id, user_id")\
            .in_("outfit_id", part).eq("is_public", True).eq("under_review", False)\
            .execute().data or []
        eligible += [str(r["outfit_id"]) for r in rows
                     if str(r["user_id"]) != me and str(r["user_id"]) not in excluded]
    if eligible:
        today = datetime.now(timezone.utc).date().isoformat()
        try:
            supabase.table("post_views").upsert(
                [{"post_id": pid, "viewer_id": me, "day": today} for pid in eligible],
                on_conflict="post_id,viewer_id,day", ignore_duplicates=True,
            ).execute()
        except Exception:
            logger.exception("record views failed viewer=%s", me)     # never break scrolling
    return {"ok": True, "counted": len(eligible)}


# ---------------------------------------------------------------- shares

def share_count(post_id: str) -> int:
    return supabase.table("post_shares").select("share_id", count="exact")\
        .eq("post_id", post_id).limit(1).execute().count or 0


def bulk_share_counts(post_ids: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    ids = [str(p) for p in post_ids if p]
    for part in _chunks(ids):
        for r in supabase.table("post_shares").select("post_id").in_("post_id", part).execute().data or []:
            pid = str(r["post_id"])
            out[pid] = out.get(pid, 0) + 1
    return out


def record_share(post_id, user_id, channel) -> dict:
    pid, me = str(post_id), str(user_id)
    channel = str(channel or "").strip().lower()
    if channel not in CHANNELS:
        raise _bad("INVALID_CHANNEL", f"channel must be one of: {', '.join(CHANNELS)}.")
    _visible_post(pid, me)                                         # 404 POST_NOT_FOUND
    supabase.table("post_shares").insert({"post_id": pid, "user_id": me, "channel": channel}).execute()
    return {"post_id": pid, "share_count": share_count(pid)}


# ---------------------------------------------------------------- insights

def _day(ts) -> str:
    return str(ts)[:10]


def post_insights(post_id, user_id, range_: str = "7d") -> dict:
    pid, me = str(post_id), str(user_id)
    days = RANGES.get(range_)
    if not days:
        raise _bad("INVALID_RANGE", "range must be 7d or 30d.")
    row = supabase.table("outfits").select("outfit_id, user_id").eq("outfit_id", pid)\
        .limit(1).execute().data
    if not row:
        raise _bad("POST_NOT_FOUND", "This post isn't available.", 404)
    if str(row[0]["user_id"]) != me:
        raise _bad("NOT_OWNER", "Only the creator can see insights for this post.", 403)

    today = datetime.now(timezone.utc).date()
    since = today - timedelta(days=days - 1)
    since_ts = datetime(since.year, since.month, since.day, tzinfo=timezone.utc).isoformat()
    daily = {(since + timedelta(days=i)).isoformat(): {"views": 0, "reactions": 0} for i in range(days)}

    views = supabase.table("post_views").select("day").eq("post_id", pid)\
        .gte("day", since.isoformat()).execute().data or []
    for v in views:
        d = _day(v["day"])
        if d in daily:
            daily[d]["views"] += 1

    by_type = {t: 0 for t in REACTION_TYPES}
    reactions = supabase.table("outfit_reactions").select("reaction, created_at")\
        .eq("outfit_id", pid).gte("created_at", since_ts).execute().data or []
    for r in reactions:
        if r.get("reaction") in by_type:
            by_type[r["reaction"]] += 1
        d = _day(r["created_at"])
        if d in daily:
            daily[d]["reactions"] += 1

    bookmarks = supabase.table("saved_outfits").select("outfit_id", count="exact")\
        .eq("outfit_id", pid).neq("user_id", me).gte("created_at", since_ts)\
        .limit(1).execute().count or 0
    collected = supabase.table("collection_entries").select("collection_id", count="exact")\
        .eq("post_id", pid).neq("user_id", me).gte("added_at", since_ts)\
        .limit(1).execute().count or 0
    shares = supabase.table("post_shares").select("share_id", count="exact")\
        .eq("post_id", pid).gte("created_at", since_ts).limit(1).execute().count or 0
    recreates = supabase.table("post_recreates").select("post_id", count="exact")\
        .eq("post_id", pid).gte("day", since.isoformat()).limit(1).execute().count or 0

    return {
        "post_id": pid,
        "range": range_,
        "from": since.isoformat(),
        "to": today.isoformat(),
        "views": len(views),
        "saves": bookmarks + collected,
        "shares": shares,
        "recreates": recreates,
        "reactions": {"total": sum(by_type.values()), "by_type": by_type},
        "daily": [{"date": d, **v} for d, v in daily.items()],
    }