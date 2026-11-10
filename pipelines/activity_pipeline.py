"""
D15 Activity - the bell (2026-11-03).

Table (see the D15 SQL):
    activity (activity_id, user_id = who sees it, type, group_key, post_id, comment_id,
              actor_ids uuid[] newest first, actor_count, reaction_type, milestone,
              read, created_at, updated_at)   unique (user_id, group_key)

Types and who gets them:
    reacted        post owner, when someone reacts to their post
    commented      post owner, when someone comments / replies on their post
    replied        comment author, when someone replies to their comment
    mentioned      a user @mentioned in a comment or (MENTION-CAPTION-1) a post caption
    followed       the person followed
    liked_comment  comment author, when someone likes their comment
    milestone      post owner, when a post reaches MILESTONES reactions

Grouping: one row per (type, target, UTC day) - target = the post (the comment
for liked_comment, nothing for followed). Exception - mentioned: one row per
(person, comment) or (person, caption), no day, created once and never bumped,
so editing a caption / comment can't notify the same person twice. A new event updates the row: actor
moved to the front, count +1 if new, updated_at = now, read = false. Milestones
are one row per (post, value), never repeated.

Never: yourself, people blocked either way. Hidden/blocked actors are filtered
on read. Recording never breaks the action that caused it (errors are logged).

PUSH-1 (2026-11-06): when a row is created, or gets a NEW actor, it is handed to
push_pipeline.enqueue (settings, reaction digest, quiet hours live there). The
same person reacting again, or a mention row that already exists, pushes nothing.
"""

import logging
from datetime import datetime, timezone

from postgrest.exceptions import APIError

from core.storage.supabase_client import supabase
from pipelines.safety_pipeline import excluded_user_ids, is_blocked_either_way
from pipelines.push_pipeline import enqueue as enqueue_push

logger = logging.getLogger(__name__)

TYPES = ("reacted", "commented", "replied", "mentioned", "followed", "liked_comment", "milestone")
MAX_ACTORS_KEPT = 20
SHOWN_ACTORS = 2
MILESTONES = (10, 50, 100, 500, 1000)
PREVIEW_LEN = 80
PREVIEW_PIECES = 3
DEFAULT_PAGE = 20
MAX_PAGE = 50


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------- write

def _upsert(recipient: str, actor: str | None, type_: str, group_key: str, *,
            post_id=None, comment_id=None, reaction_type=None, milestone=None,
            once: bool = False, _retry: bool = True) -> str | None:
    """Returns the activity_id when there's something NEW to push about
    (new row, or a new actor on an existing row), else None."""
    res = supabase.table("activity").select("activity_id, actor_ids, actor_count")\
        .eq("user_id", recipient).eq("group_key", group_key).limit(1).execute()
    if res.data and once:
        return None                                    # already notified
    if res.data:
        row = res.data[0]
        ids = [str(i) for i in (row.get("actor_ids") or [])]
        count = int(row.get("actor_count") or 0)
        new_actor = False
        if actor:
            if actor not in ids:
                count += 1
                new_actor = True
            ids = ([actor] + [i for i in ids if i != actor])[:MAX_ACTORS_KEPT]
        update = {"actor_ids": ids, "actor_count": count, "updated_at": _now(), "read": False}
        if comment_id:
            update["comment_id"] = str(comment_id)
        if reaction_type:
            update["reaction_type"] = reaction_type
        supabase.table("activity").update(update).eq("activity_id", row["activity_id"]).execute()
        return str(row["activity_id"]) if new_actor else None
    try:
        ins = supabase.table("activity").insert({
            "user_id": recipient, "type": type_, "group_key": group_key,
            "post_id": str(post_id) if post_id else None,
            "comment_id": str(comment_id) if comment_id else None,
            "actor_ids": [actor] if actor else [],
            "actor_count": 1 if actor else 0,
            "reaction_type": reaction_type, "milestone": milestone,
        }).execute()
        return str(ins.data[0]["activity_id"]) if ins.data else None
    except APIError as exc:
        if getattr(exc, "code", None) == "23505" and _retry:     # created at the same moment
            return _upsert(recipient, actor, type_, group_key, post_id=post_id, comment_id=comment_id,
                           reaction_type=reaction_type, milestone=milestone, once=once, _retry=False)
        raise


def record(recipient, actor, type_: str, *, post_id=None, comment_id=None,
           reaction_type=None) -> None:
    """Best effort: never raises."""
    try:
        recipient, actor = str(recipient or ""), str(actor or "")
        if not recipient or not actor or recipient == actor:
            return
        if is_blocked_either_way(recipient, actor):
            return
        if type_ == "mentioned":
            key = f"mentioned:comment:{comment_id}" if comment_id else f"mentioned:caption:{post_id}"
            changed = _upsert(recipient, actor, type_, key, post_id=post_id, comment_id=comment_id, once=True)
        else:
            day = datetime.now(timezone.utc).date().isoformat()
            target = comment_id if type_ == "liked_comment" else (post_id or "")
            changed = _upsert(recipient, actor, type_, f"{type_}:{target}:{day}",
                              post_id=post_id, comment_id=comment_id, reaction_type=reaction_type)
        if changed:
            enqueue_push(recipient, changed, type_)               # PUSH-1
    except Exception:
        logger.exception("activity record failed type=%s to=%s", type_, recipient)


def record_reaction(owner, actor, post_id, reaction_type: str, total_reactions: int) -> None:
    record(owner, actor, "reacted", post_id=post_id, reaction_type=reaction_type)
    if total_reactions in MILESTONES:
        try:
            changed = _upsert(str(owner), None, "milestone", f"milestone:{post_id}:{total_reactions}",
                              post_id=post_id, milestone={"kind": "reactions", "value": total_reactions},
                              once=True)
            if changed:
                enqueue_push(str(owner), changed, "milestone")    # PUSH-1
        except Exception:
            logger.exception("activity milestone failed post=%s", post_id)


def record_comment(comment: dict, post_owner, parent_author=None) -> None:
    """commented -> post owner, replied -> parent author, mentioned -> each @mention."""
    actor = str(comment["user"]["user_id"]) if comment.get("user") else None
    pid, cid = comment["post_id"], comment["comment_id"]
    notified = {actor}
    if parent_author and str(parent_author) not in notified:
        record(parent_author, actor, "replied", post_id=pid, comment_id=cid)
        notified.add(str(parent_author))
    if post_owner and str(post_owner) not in notified:
        record(post_owner, actor, "commented", post_id=pid, comment_id=cid)
        notified.add(str(post_owner))
    for m in comment.get("mentions") or []:
        uid = str(m.get("user_id"))
        if uid not in notified:
            record(uid, actor, "mentioned", post_id=pid, comment_id=cid)
            notified.add(uid)


# ---------------------------------------------------------------- read

def _preview(text: str | None) -> str | None:
    text = text or ""
    if not text:
        return None
    return text if len(text) <= PREVIEW_LEN else text[:PREVIEW_LEN - 1] + "\u2026"


def list_activity(user_id, cursor: str | None = None, limit: int = DEFAULT_PAGE) -> dict:
    # Imported here to avoid circular imports.
    from pipelines.users_pipeline import build_users, _encode_cursor, _decode_cursor
    from api.social import _get_bulk_item_details

    me = str(user_id)
    limit = max(1, min(int(limit), MAX_PAGE))
    q = supabase.table("activity").select("*").eq("user_id", me)
    if cursor:
        ts, aid = _decode_cursor(cursor)
        q = q.or_(f'updated_at.lt."{ts}",and(updated_at.eq."{ts}",activity_id.lt.{aid})')
    rows = q.order("updated_at", desc=True).order("activity_id", desc=True)\
        .limit(limit + 1).execute().data or []
    has_more = len(rows) > limit
    rows = rows[:limit]
    next_cursor = _encode_cursor(str(rows[-1]["updated_at"]), str(rows[-1]["activity_id"])) \
        if has_more and rows else None

    excluded = excluded_user_ids(me)
    post_ids = list({str(r["post_id"]) for r in rows if r.get("post_id")})
    comment_ids = list({str(r["comment_id"]) for r in rows if r.get("comment_id")})

    posts = {}
    if post_ids:
        for p in supabase.table("outfits").select("outfit_id, user_id, is_public, item_ids, caption")\
                .in_("outfit_id", post_ids).execute().data or []:
            posts[str(p["outfit_id"])] = p
    pieces = _get_bulk_item_details(list({i for p in posts.values()
                                          for i in (p.get("item_ids") or [])[:PREVIEW_PIECES]}))
    comments = {}
    if comment_ids:
        for c in supabase.table("comments").select("comment_id, text, deleted")\
                .in_("comment_id", comment_ids).execute().data or []:
            comments[str(c["comment_id"])] = c

    visible_actors = {}
    for r in rows:
        visible_actors[r["activity_id"]] = [str(a) for a in (r.get("actor_ids") or [])
                                            if str(a) not in excluded]
    people = build_users([a for ids in visible_actors.values() for a in ids[:SHOWN_ACTORS]], me)

    items = []
    for r in rows:
        actors = visible_actors[r["activity_id"]]
        if r["type"] != "milestone" and not actors:
            continue                                   # everyone in it is blocked / hidden
        post = None
        if r.get("post_id"):
            p = posts.get(str(r["post_id"]))
            if not p or (not p.get("is_public") and str(p.get("user_id")) != me):
                continue                               # post gone / unpublished
            post = {"post_id": str(p["outfit_id"]),
                    "items": [pieces.get(i, {"item_id": i, "deleted": True})
                              for i in (p.get("item_ids") or [])[:PREVIEW_PIECES]]}
        comment = None
        text_preview = None
        if r.get("comment_id"):
            c = comments.get(str(r["comment_id"]))
            if not c or c.get("deleted"):
                continue                               # comment removed
            text_preview = _preview(c.get("text"))
            comment = {"comment_id": str(c["comment_id"]), "text_preview": text_preview or ""}
        elif r["type"] == "mentioned" and post:
            # Caption mention: preview the caption. Removed from the caption -> still shown.
            text_preview = _preview(posts[str(r["post_id"])].get("caption"))
        hidden_count = len([a for a in (r.get("actor_ids") or []) if str(a) in excluded])
        shown = [people[a] for a in actors[:SHOWN_ACTORS] if a in people]
        items.append({
            "activity_id": str(r["activity_id"]),
            "type": r["type"],
            "actors": shown,
            "actor": shown[0] if shown else None,          # MENTION-CAPTION-1 (= actors[0])
            "actor_count": max(len(actors), int(r.get("actor_count") or 0) - hidden_count),
            "post": post,
            "comment": comment,
            "post_id": post["post_id"] if post else None,
            "comment_id": comment["comment_id"] if comment else None,
            "text_preview": text_preview,
            "reaction_type": r.get("reaction_type"),
            "milestone": r.get("milestone"),
            "created_at": r["updated_at"],
            "read": bool(r.get("read")),
        })
    return {"items": items, "next_cursor": next_cursor}


def unread_count(user_id) -> dict:
    n = supabase.table("activity").select("activity_id", count="exact")\
        .eq("user_id", str(user_id)).eq("read", False).limit(1).execute().count or 0
    return {"unread_count": n}


def mark_read(user_id, up_to_activity_id=None) -> dict:
    me = str(user_id)
    q = supabase.table("activity").update({"read": True}).eq("user_id", me).eq("read", False)
    if up_to_activity_id:
        row = supabase.table("activity").select("updated_at").eq("user_id", me)\
            .eq("activity_id", str(up_to_activity_id)).limit(1).execute().data
        if not row:
            return {**unread_count(me), "marked": 0}
        q = q.lte("updated_at", row[0]["updated_at"])
    marked = len(q.execute().data or [])
    return {**unread_count(me), "marked": marked}