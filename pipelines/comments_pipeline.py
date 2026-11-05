"""
C13 Comments (2026-11-03).

Tables (see the C13 SQL):
    comments       (comment_id, post_id, parent_id, user_id, text, mentions jsonb,
                    pinned, deleted, created_at, edited_at)
    comment_likes  (comment_id, user_id, created_at)
    outfits.comments_enabled (bool, default true)

Rules:
    * text 1-500 chars (trimmed). One level of replies: replying to a reply
      attaches to that reply's top-level comment.
    * author can edit the text for EDIT_WINDOW_MIN minutes (edited: true).
    * author OR post owner can delete. Soft delete: text/mentions/user hidden,
      replies stay. A deleted comment with no visible replies is not listed.
    * post owner can pin ONE top-level comment (pinning another unpins it).
    * @handles are parsed on write into mentions [{user_id, handle, start, end}],
      start/end in UTF-16 code units (Dart string indexes).
    * rate limit: RATE_PER_MIN per minute, RATE_PER_DAY per day -> 429 + Retry-After.
    * comments by people blocked either way / hidden by me are never shown.
    * comments_enabled=false: list still works (read-only), new comments 403 COMMENTS_OFF.
    * no automatic text filter yet - reports (E18) cover abuse.
    * D15: new comments / replies / @mentions / new likes go to the bell.
"""

import base64
import logging
import re
from datetime import datetime, timedelta, timezone
from uuid import UUID

from core.storage.supabase_client import supabase
from shared.exceptions import FitCheckException
from pipelines.safety_pipeline import (
    _visible_post, excluded_user_ids, is_blocked_either_way, not_in,
)
from pipelines.activity_pipeline import record, record_comment

logger = logging.getLogger(__name__)

MAX_TEXT = 500
EDIT_WINDOW_MIN = 15
RATE_PER_MIN = 10
RATE_PER_DAY = 200
INLINE_REPLIES = 2
TOP_COMMENTS = 2
DEFAULT_PAGE = 20
MAX_PAGE = 50
RELEVANT_CAP = 200          # relevant sort ranks at most this many top-level comments
MENTION_RE = re.compile(r"(?<![\w@])@([A-Za-z0-9_.]{3,20})")

COMMENT_FIELDS = "comment_id, post_id, parent_id, user_id, text, mentions, pinned, deleted, created_at, edited_at"


# ---------------------------------------------------------------- errors

def _bad(code: str, message: str, status: int = 400) -> FitCheckException:
    return FitCheckException(message, code=code, status_code=status)


def _comment_not_found() -> FitCheckException:
    return _bad("COMMENT_NOT_FOUND", "This comment isn't available.", 404)


def _not_allowed(msg: str = "You can't do that.") -> FitCheckException:
    return _bad("NOT_ALLOWED", msg, 403)


def _rate_limited(seconds: int) -> FitCheckException:
    exc = _bad("RATE_LIMITED", "You're commenting too fast. Try again in a moment.", 429)
    exc.headers = {"Retry-After": str(seconds)}
    return exc


# ---------------------------------------------------------------- helpers

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(value) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _utf16_len(s: str) -> int:
    return len(s.encode("utf-16-le")) // 2


def _clean_text(raw) -> str:
    text = str(raw or "").strip()
    if not text:
        raise _bad("COMMENT_EMPTY", "Write something first.")
    if len(text) > MAX_TEXT:
        raise _bad("COMMENT_TOO_LONG", f"Comments can be up to {MAX_TEXT} characters.")
    return text


def _parse_mentions(text: str) -> list[dict]:
    found = []
    for m in MENTION_RE.finditer(text):
        handle = m.group(1).rstrip(".").lower()
        if len(handle) < 3:
            continue
        start = m.start()
        end = start + 1 + len(handle)               # '@' + handle (trailing dots excluded)
        found.append((handle, start, end))
    if not found:
        return []
    handles = list({h for h, _, _ in found})
    rows = supabase.table("profiles").select("id, handle").in_("handle", handles).execute().data or []
    owner = {r["handle"]: str(r["id"]) for r in rows}
    out = []
    for handle, start, end in found:
        if handle in owner:
            out.append({
                "user_id": owner[handle], "handle": handle,
                "start": _utf16_len(text[:start]), "end": _utf16_len(text[:end]),
            })
    return out


def _encode(kind: str, *parts) -> str:
    return base64.urlsafe_b64encode("|".join([kind, *map(str, parts)]).encode()).decode()


def _decode(cursor: str, kind: str) -> list[str]:
    try:
        parts = base64.urlsafe_b64decode(cursor.encode()).decode().split("|")
        if parts[0] != kind:
            raise ValueError
        return parts[1:]
    except Exception:
        raise _bad("INVALID_CURSOR", "That page cursor isn't valid. Start from the first page.")


def _get_comment(comment_id) -> dict:
    res = supabase.table("comments").select(COMMENT_FIELDS)\
        .eq("comment_id", str(comment_id)).limit(1).execute()
    if not res.data:
        raise _comment_not_found()
    return res.data[0]


def _post_for_comment(comment: dict, viewer: str) -> dict:
    """The comment's post, visible to the viewer; else 404 COMMENT_NOT_FOUND."""
    try:
        post = _visible_post(str(comment["post_id"]), viewer)
    except FitCheckException:
        raise _comment_not_found()
    author = comment.get("user_id")
    if author and not comment.get("deleted") and is_blocked_either_way(author, viewer):
        raise _comment_not_found()
    return post


def _post_row(post_id: str) -> dict:
    res = supabase.table("outfits").select("outfit_id, user_id, comments_enabled")\
        .eq("outfit_id", post_id).limit(1).execute()
    return res.data[0] if res.data else {}


# ---------------------------------------------------------------- building objects

def _like_info(comment_ids: list[str], viewer: str) -> tuple[dict[str, int], set[str]]:
    if not comment_ids:
        return {}, set()
    rows = supabase.table("comment_likes").select("comment_id, user_id")\
        .in_("comment_id", comment_ids).execute().data or []
    counts: dict[str, int] = {}
    mine = set()
    for r in rows:
        cid = str(r["comment_id"])
        counts[cid] = counts.get(cid, 0) + 1
        if str(r["user_id"]) == viewer:
            mine.add(cid)
    return counts, mine


def _replies_for(parent_ids: list[str], excluded: set[str]) -> dict[str, list[dict]]:
    """parent_id -> visible replies, oldest first."""
    if not parent_ids:
        return {}
    q = supabase.table("comments").select(COMMENT_FIELDS)\
        .in_("parent_id", parent_ids).eq("deleted", False)
    q = not_in(q, "user_id", excluded)
    rows = q.order("created_at").order("comment_id").execute().data or []
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(str(r["parent_id"]), []).append(r)
    return out


def _build(rows: list[dict], viewer: str, post_owner_by_post: dict[str, str],
           replies_by_parent: dict[str, list[dict]] | None = None,
           inline: int = 0) -> list[dict]:
    """Rows -> comment objects. inline>0 adds up to `inline` replies to each."""
    from pipelines.users_pipeline import build_users, unknown_user   # avoid circular import

    replies_by_parent = replies_by_parent or {}
    inline_rows = [rep for r in rows for rep in replies_by_parent.get(str(r["comment_id"]), [])[:inline]]
    all_rows = rows + inline_rows
    ids = [str(r["comment_id"]) for r in all_rows]
    like_counts, my_likes = _like_info(ids, viewer)
    people = build_users([r["user_id"] for r in all_rows if r.get("user_id") and not r.get("deleted")], viewer)

    def one(r: dict, with_replies: bool) -> dict:
        cid = str(r["comment_id"])
        deleted = bool(r.get("deleted"))
        uid = str(r["user_id"]) if r.get("user_id") else None
        owner = post_owner_by_post.get(str(r["post_id"]))
        reps = replies_by_parent.get(cid, [])
        obj = {
            "comment_id": cid,
            "post_id": str(r["post_id"]),
            "parent_id": str(r["parent_id"]) if r.get("parent_id") else None,
            "user": None if deleted or not uid else (people.get(uid) or unknown_user(uid, viewer)),
            "text": None if deleted else r.get("text"),
            "mentions": [] if deleted else (r.get("mentions") or []),
            "created_at": r["created_at"],
            "edited": bool(r.get("edited_at")) and not deleted,
            "like_count": like_counts.get(cid, 0),
            "liked_by_me": cid in my_likes,
            "reply_count": len(reps) if not r.get("parent_id") else 0,
            "replies": [],
            "is_creator": bool(uid) and uid == owner and not deleted,
            "pinned": bool(r.get("pinned")) and not deleted,
            "deleted": deleted,
        }
        if with_replies:
            obj["replies"] = [one(rep, False) for rep in reps[:inline]]
        return obj

    return [one(r, inline > 0) for r in rows]


# ---------------------------------------------------------------- list

def list_comments(post_id, viewer_id, sort: str = "relevant",
                  cursor: str | None = None, limit: int = DEFAULT_PAGE) -> dict:
    pid, viewer = str(post_id), str(viewer_id)
    post = _visible_post(pid, viewer)
    owner = str(post["user_id"])
    limit = max(1, min(int(limit), MAX_PAGE))
    excluded = excluded_user_ids(viewer)

    def top_level():
        # A fresh query each time: postgrest builders change in place.
        q = supabase.table("comments").select(COMMENT_FIELDS)\
            .eq("post_id", pid).is_("parent_id", "null")
        return not_in(q, "user_id", excluded)

    if sort == "newest":
        q = top_level()
        if cursor:
            ts, cid = _decode(cursor, "t")
            UUID(cid)
            q = q.or_(f'created_at.lt."{ts}",and(created_at.eq."{ts}",comment_id.lt.{cid})')
        q = q.eq("pinned", False)
        rows = q.order("created_at", desc=True).order("comment_id", desc=True)\
            .limit(limit + 1).execute().data or []
        has_more = len(rows) > limit
        rows = rows[:limit]
        if not cursor:                                      # pinned first, first page only
            pinned = top_level().eq("pinned", True).limit(1).execute().data or []
            rows = pinned + rows
        last = rows[-1] if rows else None
        next_cursor = _encode("t", last["created_at"], last["comment_id"]) if has_more and last else None
    else:                                                   # relevant
        offset = 0
        if cursor:
            try:
                offset = max(0, int(_decode(cursor, "o")[0]))
            except ValueError:
                raise _bad("INVALID_CURSOR", "That page cursor isn't valid. Start from the first page.")
        everything = top_level().order("created_at", desc=True).limit(RELEVANT_CAP).execute().data or []
        likes, _ = _like_info([str(r["comment_id"]) for r in everything], viewer)
        everything.sort(key=lambda r: (
            not r.get("pinned"),
            str(r.get("user_id")) != owner,
            -likes.get(str(r["comment_id"]), 0),
            -_parse_ts(r["created_at"]).timestamp(),
        ))
        rows = everything[offset:offset + limit]
        has_more = offset + limit < len(everything)
        next_cursor = _encode("o", offset + limit) if has_more else None

    replies = _replies_for([str(r["comment_id"]) for r in rows], excluded)
    # A deleted top-level comment only stays if it still has visible replies.
    rows = [r for r in rows if not r.get("deleted") or replies.get(str(r["comment_id"]))]
    items = _build(rows, viewer, {pid: owner}, replies, INLINE_REPLIES)

    return {"items": items, "next_cursor": next_cursor,
            "count": comment_count(pid, excluded),
            "comments_enabled": post_comments_enabled(pid)}


def list_replies(comment_id, viewer_id, cursor: str | None = None, limit: int = DEFAULT_PAGE) -> dict:
    viewer = str(viewer_id)
    parent = _get_comment(comment_id)
    post = _post_for_comment(parent, viewer)
    limit = max(1, min(int(limit), MAX_PAGE))
    excluded = excluded_user_ids(viewer)

    q = supabase.table("comments").select(COMMENT_FIELDS)\
        .eq("parent_id", str(parent["comment_id"])).eq("deleted", False)
    q = not_in(q, "user_id", excluded)
    if cursor:
        ts, cid = _decode(cursor, "r")
        UUID(cid)
        q = q.or_(f'created_at.gt."{ts}",and(created_at.eq."{ts}",comment_id.gt.{cid})')
    rows = q.order("created_at").order("comment_id").limit(limit + 1).execute().data or []
    has_more = len(rows) > limit
    rows = rows[:limit]
    items = _build(rows, viewer, {str(post["outfit_id"]): str(post["user_id"])})
    last = rows[-1] if rows else None
    return {"items": items,
            "next_cursor": _encode("r", last["created_at"], last["comment_id"]) if has_more and last else None}


def comment_count(post_id: str, excluded: set[str]) -> int:
    q = supabase.table("comments").select("comment_id", count="exact")\
        .eq("post_id", post_id).eq("deleted", False)
    q = not_in(q, "user_id", excluded)
    return q.limit(1).execute().count or 0


def post_comments_enabled(post_id: str) -> bool:
    row = _post_row(post_id)
    return row.get("comments_enabled") is not False


# ---------------------------------------------------------------- feed items

def bulk_comment_summary(posts: list[dict], viewer_id: str,
                         excluded: set[str]) -> tuple[dict[str, int], dict[str, list[dict]]]:
    """For feed items: post_id -> comment_count, post_id -> top_comments (pinned first, then newest)."""
    post_ids = [str(p["outfit_id"]) for p in posts]
    if not post_ids:
        return {}, {}
    q = supabase.table("comments").select(COMMENT_FIELDS)\
        .in_("post_id", post_ids).eq("deleted", False)
    q = not_in(q, "user_id", excluded)
    rows = q.order("created_at", desc=True).execute().data or []

    counts: dict[str, int] = {pid: 0 for pid in post_ids}
    tops: dict[str, list[dict]] = {pid: [] for pid in post_ids}
    replies: dict[str, list[dict]] = {}
    for r in rows:
        pid = str(r["post_id"])
        counts[pid] = counts.get(pid, 0) + 1
        if r.get("parent_id"):
            replies.setdefault(str(r["parent_id"]), []).append(r)
        else:
            tops.setdefault(pid, []).append(r)

    chosen: list[dict] = []
    for pid, top in tops.items():
        top.sort(key=lambda r: (not r.get("pinned"), -_parse_ts(r["created_at"]).timestamp()))
        chosen.extend(top[:TOP_COMMENTS])
    owners = {str(p["outfit_id"]): str(p["user_id"]) for p in posts}
    built = _build(chosen, str(viewer_id), owners, replies, inline=0)
    out: dict[str, list[dict]] = {pid: [] for pid in post_ids}
    for c in built:
        out[c["post_id"]].append(c)
    return counts, out


# ---------------------------------------------------------------- write

def _check_rate(uid: str) -> None:
    now = _now()
    minute = supabase.table("comments").select("comment_id", count="exact")\
        .eq("user_id", uid).gte("created_at", (now - timedelta(minutes=1)).isoformat())\
        .limit(1).execute().count or 0
    if minute >= RATE_PER_MIN:
        raise _rate_limited(60)
    day = supabase.table("comments").select("comment_id", count="exact")\
        .eq("user_id", uid).gte("created_at", (now - timedelta(days=1)).isoformat())\
        .limit(1).execute().count or 0
    if day >= RATE_PER_DAY:
        raise _rate_limited(3600)


def create_comment(post_id, user_id, text, parent_id=None) -> dict:
    pid, me = str(post_id), str(user_id)
    clean = _clean_text(text)
    post = _visible_post(pid, me)
    if post_comments_enabled(pid) is False:
        raise _bad("COMMENTS_OFF", "Comments are turned off for this post.", 403)

    parent = None
    if parent_id:
        parent = _get_comment(parent_id)
        if str(parent["post_id"]) != pid:
            raise _comment_not_found()
        if parent.get("parent_id"):                     # one level only
            parent = _get_comment(parent["parent_id"])
        if parent.get("user_id") and is_blocked_either_way(parent["user_id"], me):
            raise _comment_not_found()

    _check_rate(me)
    row = supabase.table("comments").insert({
        "post_id": pid,
        "parent_id": str(parent["comment_id"]) if parent else None,
        "user_id": me,
        "text": clean,
        "mentions": _parse_mentions(clean),
    }).execute().data[0]
    logger.info("comment created post=%s user=%s parent=%s", pid, me, parent and parent["comment_id"])
    comment = _build([row], me, {pid: str(post["user_id"])})[0]
    record_comment(comment, post["user_id"], parent.get("user_id") if parent else None)   # D15
    return comment


def edit_comment(comment_id, user_id, sent: dict) -> dict:
    me = str(user_id)
    c = _get_comment(comment_id)
    post = _post_for_comment(c, me)
    owner = str(post["user_id"])
    if c.get("deleted"):
        raise _comment_not_found()

    if "text" in sent:
        if str(c.get("user_id")) != me:
            raise _not_allowed("Only the author can edit a comment.")
        if _now() - _parse_ts(c["created_at"]) > timedelta(minutes=EDIT_WINDOW_MIN):
            raise _bad("EDIT_WINDOW_CLOSED",
                       f"Comments can only be edited for {EDIT_WINDOW_MIN} minutes.", 403)
        clean = _clean_text(sent["text"])
        supabase.table("comments").update({
            "text": clean, "mentions": _parse_mentions(clean), "edited_at": _now().isoformat(),
        }).eq("comment_id", c["comment_id"]).execute()

    if "pinned" in sent and sent["pinned"] is not None:
        if owner != me:
            raise _not_allowed("Only the post's creator can pin comments.")
        if c.get("parent_id"):
            raise _not_allowed("Only top-level comments can be pinned.")
        if sent["pinned"]:
            supabase.table("comments").update({"pinned": False})\
                .eq("post_id", c["post_id"]).eq("pinned", True).execute()
        supabase.table("comments").update({"pinned": bool(sent["pinned"])})\
            .eq("comment_id", c["comment_id"]).execute()

    fresh = _get_comment(c["comment_id"])
    replies = _replies_for([str(fresh["comment_id"])], excluded_user_ids(me)) if not fresh.get("parent_id") else {}
    return _build([fresh], me, {str(post["outfit_id"]): owner}, replies, INLINE_REPLIES)[0]


def soft_delete(comment_ids: list[str]) -> None:
    if comment_ids:
        supabase.table("comments").update({
            "text": None, "mentions": [], "deleted": True, "pinned": False,
        }).in_("comment_id", comment_ids).execute()


def delete_comment(comment_id, user_id) -> dict:
    me = str(user_id)
    c = _get_comment(comment_id)
    cid = str(c["comment_id"])
    if not c.get("deleted"):
        post = _post_row(str(c["post_id"]))
        if me not in (str(c.get("user_id")), str(post.get("user_id"))):
            raise _not_allowed("Only the author or the post's creator can delete this.")
        soft_delete([cid])
        logger.info("comment deleted comment=%s by=%s", cid, me)
    return {"deleted": True, "comment_id": cid}


def _like_count(cid: str) -> int:
    return supabase.table("comment_likes").select("user_id", count="exact")\
        .eq("comment_id", cid).limit(1).execute().count or 0


def like_comment(comment_id, user_id) -> dict:
    from pipelines.safety_pipeline import _insert_ignore_duplicate
    me = str(user_id)
    c = _get_comment(comment_id)
    _post_for_comment(c, me)
    if c.get("deleted"):
        raise _comment_not_found()
    cid = str(c["comment_id"])
    already = supabase.table("comment_likes").select("user_id")\
        .eq("comment_id", cid).eq("user_id", me).limit(1).execute().data
    if not already:
        _insert_ignore_duplicate("comment_likes", {"comment_id": cid, "user_id": me})
        record(c.get("user_id"), me, "liked_comment",
               post_id=str(c["post_id"]), comment_id=cid)       # D15
    return {"liked": True, "like_count": _like_count(cid), "comment_id": cid}


def unlike_comment(comment_id, user_id) -> dict:
    cid = str(comment_id)
    supabase.table("comment_likes").delete().eq("comment_id", cid).eq("user_id", str(user_id)).execute()
    return {"liked": False, "like_count": _like_count(cid), "comment_id": cid}