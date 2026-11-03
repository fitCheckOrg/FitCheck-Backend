"""
E18 Safety (2026-11-03): block, hide, not interested, reports.

Tables (see the E18 SQL):
    user_blocks          (blocker_id, blocked_id, created_at)
    user_hides           (user_id, hidden_id, created_at)
    post_not_interested  (user_id, post_id, created_at)
    reports              (report_id, reporter_id, target_type, target_id, reason, note, status, created_at)
    outfits.under_review (bool) - set when REVIEW_THRESHOLD different people report a post

Rules:
    * BLOCK works both ways: neither side sees the other anywhere (feed, profile,
      posts, lists, reactions, search, ...). Profile -> 404 USER_NOT_FOUND.
      Blocking also unfollows both ways and removes their reactions from my posts
      (and, from C13, their comments on my posts). Idempotent.
    * HIDE is one way and silent: I stop seeing them in feeds and lists.
      Their profile still opens if I go to it.
    * NOT INTERESTED removes one post from all my feeds.
    * REPORT: one per person per target; a repeat is 200. A post with
      REVIEW_THRESHOLD open reports from different people gets under_review=true
      and is hidden from everyone except its owner.
"""

import logging

from postgrest.exceptions import APIError

from core.storage.supabase_client import supabase
from shared.exceptions import FitCheckException

logger = logging.getLogger(__name__)

REPORT_REASONS = ("not_fashion", "spam", "harassment", "nudity", "someone_elses_work", "other")
MAX_NOTE_LEN = 300
REVIEW_THRESHOLD = 3


def _bad(code: str, message: str, status: int = 400) -> FitCheckException:
    return FitCheckException(message, code=code, status_code=status)


def _user_not_found() -> FitCheckException:
    return _bad("USER_NOT_FOUND", "User not found.", 404)


def _post_not_found() -> FitCheckException:
    return _bad("POST_NOT_FOUND", "This post isn't available.", 404)


def not_in(query, column: str, ids):
    """query.not.in.(ids) - skipped when ids is empty."""
    ids = [str(i) for i in ids if i]
    if not ids:
        return query
    return query.filter(column, "not.in", f"({','.join(ids)})")


# ---------------------------------------------------------------- who I can't see

def blocked_user_ids(viewer_id) -> set[str]:
    """Everyone I blocked + everyone who blocked me."""
    v = str(viewer_id)
    mine = supabase.table("user_blocks").select("blocked_id").eq("blocker_id", v).execute().data or []
    theirs = supabase.table("user_blocks").select("blocker_id").eq("blocked_id", v).execute().data or []
    return {str(r["blocked_id"]) for r in mine} | {str(r["blocker_id"]) for r in theirs}


def excluded_user_ids(viewer_id) -> set[str]:
    """Blocked either way + hidden by me. Never shown in feeds or lists."""
    v = str(viewer_id)
    hidden = supabase.table("user_hides").select("hidden_id").eq("user_id", v).execute().data or []
    return blocked_user_ids(v) | {str(r["hidden_id"]) for r in hidden}


def is_blocked_either_way(a, b) -> bool:
    a, b = str(a), str(b)
    if a == b:
        return False
    res = supabase.table("user_blocks").select("blocker_id")\
        .or_(f"and(blocker_id.eq.{a},blocked_id.eq.{b}),and(blocker_id.eq.{b},blocked_id.eq.{a})")\
        .limit(1).execute()
    return bool(res.data)


def not_interested_post_ids(viewer_id) -> set[str]:
    rows = supabase.table("post_not_interested").select("post_id")\
        .eq("user_id", str(viewer_id)).execute().data or []
    return {str(r["post_id"]) for r in rows}


# ---------------------------------------------------------------- helpers

def _require_user(uid: str) -> None:
    res = supabase.table("profiles").select("id").eq("id", uid).limit(1).execute()
    if not res.data:
        raise _user_not_found()


def _visible_post(post_id: str, viewer: str) -> dict:
    """Public, not under review (unless mine), creator not blocked either way."""
    res = supabase.table("outfits").select("outfit_id, user_id, is_public, under_review")\
        .eq("outfit_id", post_id).limit(1).execute()
    row = res.data[0] if res.data else None
    if not row or not row.get("is_public"):
        raise _post_not_found()
    owner = str(row["user_id"])
    if owner != viewer and (row.get("under_review") or is_blocked_either_way(owner, viewer)):
        raise _post_not_found()
    return row


def _insert_ignore_duplicate(table: str, row: dict) -> None:
    try:
        supabase.table(table).insert(row).execute()
    except APIError as exc:
        if getattr(exc, "code", None) != "23505":      # already there -> fine
            raise


# ---------------------------------------------------------------- block

def block_user(user_id, target_id) -> dict:
    me, them = str(user_id), str(target_id)
    if me == them:
        raise _bad("CANNOT_BLOCK_SELF", "You can't block yourself.")
    _require_user(them)

    _insert_ignore_duplicate("user_blocks", {"blocker_id": me, "blocked_id": them})

    # Unfollow both ways.
    supabase.table("follows").delete().eq("follower_id", me).eq("following_id", them).execute()
    supabase.table("follows").delete().eq("follower_id", them).eq("following_id", me).execute()

    # Remove their reactions from my posts.
    my_posts = supabase.table("outfits").select("outfit_id").eq("user_id", me).execute().data or []
    post_ids = [p["outfit_id"] for p in my_posts]
    if post_ids:
        supabase.table("outfit_reactions").delete()\
            .eq("user_id", them).in_("outfit_id", post_ids).execute()
    # C13: their comments on my posts are removed here too, once comments exist.

    logger.info("block user=%s blocked=%s", me, them)
    return {"blocked": True, "user_id": them}


def unblock_user(user_id, target_id) -> dict:
    supabase.table("user_blocks").delete()\
        .eq("blocker_id", str(user_id)).eq("blocked_id", str(target_id)).execute()
    return {"blocked": False, "user_id": str(target_id)}


def list_blocked(user_id, cursor: str | None = None, limit: int = 20) -> dict:
    # Imported here to avoid a circular import.
    from pipelines.users_pipeline import build_users, unknown_user, _encode_cursor, _decode_cursor

    me = str(user_id)
    limit = max(1, min(int(limit), 50))
    q = supabase.table("user_blocks").select("blocked_id, created_at").eq("blocker_id", me)
    if cursor:
        ts, bid = _decode_cursor(cursor)
        q = q.or_(f'created_at.lt."{ts}",and(created_at.eq."{ts}",blocked_id.lt.{bid})')
    rows = q.order("created_at", desc=True).order("blocked_id", desc=True)\
        .limit(limit + 1).execute().data or []
    has_more = len(rows) > limit
    rows = rows[:limit]
    people = build_users([r["blocked_id"] for r in rows], me)
    items = [people.get(str(r["blocked_id"])) or unknown_user(r["blocked_id"], me) for r in rows]
    next_cursor = None
    if has_more and rows:
        next_cursor = _encode_cursor(str(rows[-1]["created_at"]), str(rows[-1]["blocked_id"]))
    return {"items": items, "next_cursor": next_cursor}


# ---------------------------------------------------------------- hide

def hide_user(user_id, target_id) -> dict:
    me, them = str(user_id), str(target_id)
    if me == them:
        raise _bad("CANNOT_HIDE_SELF", "You can't hide yourself.")
    _require_user(them)
    _insert_ignore_duplicate("user_hides", {"user_id": me, "hidden_id": them})
    return {"hidden": True, "user_id": them}


def unhide_user(user_id, target_id) -> dict:
    supabase.table("user_hides").delete()\
        .eq("user_id", str(user_id)).eq("hidden_id", str(target_id)).execute()
    return {"hidden": False, "user_id": str(target_id)}


# ---------------------------------------------------------------- not interested

def mark_not_interested(user_id, post_id) -> dict:
    me, pid = str(user_id), str(post_id)
    row = _visible_post(pid, me)
    if str(row["user_id"]) == me:
        raise _bad("NOT_ALLOWED", "That's your own post.")
    _insert_ignore_duplicate("post_not_interested", {"user_id": me, "post_id": pid})
    return {"not_interested": True, "post_id": pid}


def unmark_not_interested(user_id, post_id) -> dict:
    supabase.table("post_not_interested").delete()\
        .eq("user_id", str(user_id)).eq("post_id", str(post_id)).execute()
    return {"not_interested": False, "post_id": str(post_id)}


# ---------------------------------------------------------------- reports

def _clean_report(reason, note) -> tuple[str, str | None]:
    reason = str(reason or "").strip().lower()
    if reason not in REPORT_REASONS:
        raise _bad("INVALID_REPORT_REASON",
                   f"Reason must be one of: {', '.join(REPORT_REASONS)}.")
    note = " ".join(str(note or "").split()) or None
    if note and len(note) > MAX_NOTE_LEN:
        raise _bad("NOTE_TOO_LONG", f"Note must be {MAX_NOTE_LEN} characters or fewer.")
    if reason == "other" and not note:
        raise _bad("NOTE_REQUIRED", "Tell us a little about the problem.")
    return reason, note


def _save_report(me: str, target_type: str, target_id: str, reason: str, note) -> bool:
    """True if this is a new report, False if this person already reported it."""
    try:
        supabase.table("reports").insert({
            "reporter_id": me, "target_type": target_type, "target_id": target_id,
            "reason": reason, "note": note,
        }).execute()
        return True
    except APIError as exc:
        if getattr(exc, "code", None) == "23505":
            return False
        raise


def report_post(user_id, post_id, reason, note=None) -> dict:
    me, pid = str(user_id), str(post_id)
    reason, note = _clean_report(reason, note)
    # Not _visible_post: you can still report a post after blocking its creator.
    res = supabase.table("outfits").select("outfit_id, user_id, is_public, under_review")\
        .eq("outfit_id", pid).limit(1).execute()
    row = res.data[0] if res.data else None
    if not row or not row.get("is_public"):
        raise _post_not_found()
    if str(row["user_id"]) == me:
        raise _bad("CANNOT_REPORT_SELF", "You can't report your own post.")

    if _save_report(me, "post", pid, reason, note):
        open_reports = supabase.table("reports").select("report_id", count="exact")\
            .eq("target_type", "post").eq("target_id", pid).eq("status", "open")\
            .limit(1).execute().count or 0
        if open_reports >= REVIEW_THRESHOLD and not row.get("under_review"):
            supabase.table("outfits").update({"under_review": True}).eq("outfit_id", pid).execute()
            logger.warning("post under review post=%s reports=%s", pid, open_reports)
    logger.info("report post=%s by=%s reason=%s", pid, me, reason)
    return {"reported": True}


def report_user(user_id, target_id, reason, note=None) -> dict:
    me, them = str(user_id), str(target_id)
    reason, note = _clean_report(reason, note)
    if me == them:
        raise _bad("CANNOT_REPORT_SELF", "You can't report yourself.")
    _require_user(them)
    _save_report(me, "user", them, reason, note)
    logger.info("report user=%s by=%s reason=%s", them, me, reason)
    return {"reported": True}