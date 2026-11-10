"""
NOTE-1 Fit Notes (2026-11-10). A 24-hour "today's fit" note: a clothes collage,
up to 60 characters of text, or both. Visible to the author's followers + the author.

Tables (see the NOTE-1 SQL):
    fit_notes           (id, user_id, kind outfit|text|outfit_text, text, text_bg,
                         category, source_type, source_id, pieces jsonb,
                         created_at, expires_at, deleted_at)
                         unique (user_id) where deleted_at is null  -> one live note
    fit_note_reactions  (note_id, user_id, reaction, created_at)  pk (note_id, user_id)
    function post_fit_note(...)  -> replaces my note in ONE transaction

Rules enforced here (not only in the app):
    * Live = deleted_at is null AND expires_at > now(). Checked in EVERY read and
      write. Nothing else decides visibility (a cleanup job may delete old rows).
    * Audience = the author + people who follow the author. Never in feeds,
      search, suggestions or the web.
    * Blocked either way, or hidden by the viewer -> the note doesn't exist for
      them (404 NOTE_NOT_FOUND - never says why).
    * Handle required to post (403 HANDLE_REQUIRED).
    * Pieces are a clothes-only snapshot built HERE from the caller's own
      wardrobe items (clean cut-outs only). The app never sends image URLs;
      try-on renders and the avatar are never used.
    * No view tracking. No push for a new note. Reaction pushes follow
      notification-settings like post reactions (activity type note_reacted).
    * Reactions refused on expired (410) / missing (404) notes and on your own.
"""

import base64
import logging
import os
import unicodedata
from datetime import datetime, timedelta, timezone
from uuid import UUID

from postgrest.exceptions import APIError

from core.storage.supabase_client import supabase
from shared.exceptions import FitCheckException
from pipelines.safety_pipeline import excluded_user_ids, not_in

logger = logging.getLogger(__name__)

try:                                     # pick up .env if python-dotenv is installed
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

NOTE_TTL_MINUTES = int(os.getenv("NOTE_TTL_MINUTES") or 24 * 60)
MAX_TEXT = 60
MAX_PIECES = 6
THUMB_PIECES = 4
ROW_CAP = 100
RATE_PER_HOUR = 10
TEXT_BGS = ("linen", "blush", "ink")
SOURCE_TYPES = ("plan", "saved", "dressing", "own_post")
REACTIONS = ("love", "fire", "clean", "bold", "wear")
TOP_ACTORS = 3
DEFAULT_PAGE = 30
MAX_PAGE = 50
NOTE_FIELDS = ("id, user_id, kind, text, text_bg, category, source_type, source_id, "
               "pieces, created_at, expires_at, deleted_at")
# Collage order: top / dress first, then outer, bottom, shoes, bag, accessories.
PIECE_ORDER = {"dress": 0, "top": 0, "outerwear": 1, "bottom": 2, "shoes": 3,
               "bag": 4, "accessory": 5, "accessories": 5}


# ---------------------------------------------------------------- errors

def _bad(code: str, message: str, status: int = 400) -> FitCheckException:
    return FitCheckException(message, code=code, status_code=status)


def _not_found() -> FitCheckException:
    return _bad("NOTE_NOT_FOUND", "This note isn't available.", 404)


def _expired() -> FitCheckException:
    return _bad("NOTE_EXPIRED", "This note has expired.", 410)


def _not_allowed(msg: str = "You can't do that.") -> FitCheckException:
    return _bad("NOT_ALLOWED", msg, 403)


# ---------------------------------------------------------------- helpers

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse(ts) -> datetime:
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))


def is_live(row: dict | None) -> bool:
    return bool(row) and not row.get("deleted_at") and _parse(row["expires_at"]) > _now()


def grapheme_len(text: str) -> int:
    """User-perceived characters (an emoji, flag or skin-tone emoji = 1).
    Approximation of Unicode grapheme clusters without extra packages."""
    count, join_next, ri_open = 0, False, False
    for ch in text:
        cp = ord(ch)
        if join_next:                              # char after a zero-width joiner
            join_next = False
            continue
        if cp == 0x200D:                           # ZWJ glues the next char on
            join_next = True
            continue
        if unicodedata.combining(ch) or 0xFE00 <= cp <= 0xFE0F or 0x1F3FB <= cp <= 0x1F3FF \
                or 0xE0020 <= cp <= 0xE007F:       # marks, variation selectors, skin tones, tags
            continue
        if 0x1F1E6 <= cp <= 0x1F1FF:               # regional indicators pair into one flag
            if ri_open:
                ri_open = False
                continue
            ri_open = True
        else:
            ri_open = False
        count += 1
    return count


def _clean_text(raw) -> str | None:
    text = " ".join(str(raw or "").split())
    if not text:
        return None
    if grapheme_len(text) > MAX_TEXT:
        raise _bad("TEXT_TOO_LONG", f"Notes can be up to {MAX_TEXT} characters.")
    return text


def _today_values(user_id: str) -> set[str]:
    """Today's date in the user's timezone (notification settings) and in UTC."""
    from pipelines.push_pipeline import get_settings, _zone
    tz = _zone(get_settings(user_id).get("tz"))
    now = _now()
    return {now.astimezone(tz).date().isoformat(), now.date().isoformat()}


# ---------------------------------------------------------------- pieces (clothes only)

def _snapshot(user_id: str, item_ids: list, strict: bool) -> list[dict]:
    """Snapshot the caller's OWN wardrobe items (metadata only; images resolve at
    read time so a deleted item shows as deleted). strict=True: every id must be
    the caller's and not archived (dressing room); else missing ones are skipped."""
    ids = [str(i) for i in item_ids if i]
    ids = list(dict.fromkeys(ids))                 # unique, keep order
    if not ids:
        return []
    rows = supabase.table("closet_items")\
        .select("item_id, user_id, category, item_type, dominant_color, is_archived")\
        .in_("item_id", ids).execute().data or []
    by_id = {str(r["item_id"]): r for r in rows}
    out = []
    for i in ids:
        r = by_id.get(i)
        if not r or str(r.get("user_id")) != user_id:
            if strict or r:                        # someone else's item -> never allowed
                raise _not_allowed("Those pieces aren't yours.")
            continue                               # deleted since -> skip
        if strict and r.get("is_archived"):
            raise _not_allowed("That piece is archived.")
        out.append({"item_id": i, "category": r.get("category"),
                    "item_type": r.get("item_type"), "dominant_color": r.get("dominant_color")})
    out.sort(key=lambda p: PIECE_ORDER.get(str(p.get("category") or "").lower(), 6))
    return out[:MAX_PIECES]


def _from_source(user_id: str, source_type, source_id, item_ids) -> tuple[list[dict], str | None]:
    """(pieces, category) from the source the app names. NOT_ALLOWED if not mine."""
    st = str(source_type or "").lower()
    if st not in SOURCE_TYPES:
        raise _bad("SOURCE_INVALID", "source_type must be plan, saved, dressing or own_post.")
    if st == "dressing":
        ids = [str(i) for i in (item_ids or [])]
        if not 1 <= len(ids) <= 2:
            raise _bad("SOURCE_INVALID", "Send 1-2 item_ids for the dressing room.")
        return _snapshot(user_id, ids, strict=True), None

    try:
        sid = str(UUID(str(source_id)))
    except Exception:
        raise _bad("SOURCE_INVALID", "source_id is missing or not valid.")

    if st == "plan":
        rows = supabase.table("planned_outfits")\
            .select("planned_id, user_id, outfit_id, item_ids, scheduled_date")\
            .eq("planned_id", sid).limit(1).execute().data
        plan = rows[0] if rows else None
        if not plan or str(plan["user_id"]) != user_id:
            raise _not_allowed("That plan isn't yours.")
        if str(plan.get("scheduled_date"))[:10] not in _today_values(user_id):
            raise _bad("PLAN_NOT_TODAY", "Only today's plan can be shared as a note.")
        category = None
        if plan.get("outfit_id"):
            o = supabase.table("outfits").select("category, user_id")\
                .eq("outfit_id", str(plan["outfit_id"])).limit(1).execute().data
            if o and str(o[0].get("user_id")) == user_id:
                category = o[0].get("category")
        return _snapshot(user_id, plan.get("item_ids") or [], strict=False), category

    rows = supabase.table("outfits").select("outfit_id, user_id, item_ids, category, is_public")\
        .eq("outfit_id", sid).limit(1).execute().data
    outfit = rows[0] if rows else None
    if not outfit or str(outfit["user_id"]) != user_id:
        raise _not_allowed("That outfit isn't yours.")
    if st == "own_post" and not outfit.get("is_public"):
        raise _not_allowed("That post isn't published.")
    return _snapshot(user_id, outfit.get("item_ids") or [], strict=False), outfit.get("category")


def _resolve_pieces(notes: list[dict], limit: int = MAX_PIECES) -> dict[str, list[dict]]:
    """note id -> pieces with fresh cut-out urls; deleted items -> deleted: true."""
    from api.social import _get_bulk_item_details        # clean_image_url only
    ids = list({str(p["item_id"]) for n in notes for p in (n.get("pieces") or [])[:limit]})
    details = _get_bulk_item_details(ids)
    out = {}
    for n in notes:
        pieces = []
        for p in (n.get("pieces") or [])[:limit]:
            d = details.get(str(p["item_id"]))
            pieces.append({
                "item_id": str(p["item_id"]),
                "category": (d or p).get("category"),
                "item_type": (d or p).get("item_type"),
                "dominant_color": (d or p).get("dominant_color"),
                "image_url": d.get("image_url") if d else None,
                "deleted": not d,
            })
        out[str(n["id"])] = pieces
    return out


# ---------------------------------------------------------------- visibility

def _follows(viewer: str, author: str) -> bool:
    res = supabase.table("follows").select("follower_id")\
        .eq("follower_id", viewer).eq("following_id", author).limit(1).execute()
    return bool(res.data)


def _can_see(author: str, viewer: str) -> bool:
    """Author, or a follower the author isn't blocked from / hidden by the viewer."""
    if author == viewer:
        return True
    if author in excluded_user_ids(viewer):
        return False
    return _follows(viewer, author)


def _get_row(note_id) -> dict | None:
    try:
        nid = str(UUID(str(note_id)))
    except Exception:
        return None
    rows = supabase.table("fit_notes").select(NOTE_FIELDS).eq("id", nid).limit(1).execute().data
    return rows[0] if rows else None


def _visible_row(note_id, viewer: str, allow_expired: bool = False) -> dict:
    """404 if missing / deleted / replaced / not allowed (never says which).
    410 if expired (only told to people who could have seen it)."""
    row = _get_row(note_id)
    if not row or row.get("deleted_at") or not _can_see(str(row["user_id"]), viewer):
        raise _not_found()
    if not allow_expired and not is_live(row):
        raise _expired()
    return row


def _live_query():
    return supabase.table("fit_notes").select(NOTE_FIELDS)\
        .is_("deleted_at", "null").gt("expires_at", _iso(_now()))


# ---------------------------------------------------------------- shaping

def _shape(rows: list[dict], viewer: str) -> list[dict]:
    from pipelines.users_pipeline import build_users, unknown_user
    if not rows:
        return []
    pieces = _resolve_pieces(rows)
    people = build_users([r["user_id"] for r in rows], viewer)
    ids = [str(r["id"]) for r in rows]
    mine = {}
    reacts = supabase.table("fit_note_reactions").select("note_id, reaction")\
        .eq("user_id", viewer).in_("note_id", ids).execute().data or []
    for r in reacts:
        mine[str(r["note_id"])] = r["reaction"]

    out = []
    for r in rows:
        nid, author = str(r["id"]), str(r["user_id"])
        u = people.get(author) or unknown_user(author, viewer)
        note = {
            "note_id": nid,
            "kind": r["kind"],
            "text": r.get("text"),
            "text_bg": r.get("text_bg") if r["kind"] == "text" else None,
            "category": r.get("category") if r["kind"] != "text" else None,
            "pieces": pieces.get(nid, []),
            "author": {"user_id": u["user_id"], "display_name": u.get("display_name"),
                       "handle": u.get("handle"), "avatar_url_small": u.get("avatar_url_small"),
                       "is_me": author == viewer},
            "created_at": r["created_at"],
            "expires_at": r["expires_at"],
            "my_reaction": None if author == viewer else mine.get(nid),
        }
        if author == viewer:
            note["reaction_summary"] = _reaction_summary(nid, viewer)
        out.append(note)
    return out


def _reaction_summary(note_id: str, owner: str) -> dict:
    from pipelines.users_pipeline import build_users
    q = supabase.table("fit_note_reactions").select("user_id", count="exact").eq("note_id", note_id)
    q = not_in(q, "user_id", excluded_user_ids(owner))
    rows = q.order("created_at", desc=True).limit(TOP_ACTORS).execute()
    people = build_users([r["user_id"] for r in (rows.data or [])], owner)
    return {"total": rows.count or 0,
            "top_actors": [people[str(r["user_id"])] for r in (rows.data or []) if str(r["user_id"]) in people]}


def note_thumb(row: dict) -> dict:
    """Small preview for activity / profiles (first 4 pieces, clothes only)."""
    pieces = _resolve_pieces([row], THUMB_PIECES).get(str(row["id"]), [])
    thumb = {"kind": row["kind"], "category": row.get("category") if row["kind"] != "text" else None,
             "pieces": pieces}
    if row["kind"] == "text":
        thumb["text_bg"] = row.get("text_bg")
    if row.get("text"):
        thumb["text"] = row["text"]
    return thumb


# ---------------------------------------------------------------- read

def notes_row(viewer_id) -> dict:
    me = str(viewer_id)
    following = supabase.table("follows").select("following_id")\
        .eq("follower_id", me).execute().data or []
    excluded = excluded_user_ids(me)
    authors = [str(f["following_id"]) for f in following if str(f["following_id"]) not in excluded]
    items = []
    if authors:
        rows = _live_query().in_("user_id", authors)\
            .order("created_at", desc=True).limit(ROW_CAP).execute().data or []
        items = _shape(rows, me)
    mine_rows = _live_query().eq("user_id", me).limit(1).execute().data or []
    mine = _shape(mine_rows, me)
    return {"mine": mine[0] if mine else None, "items": items}


def get_note(note_id, viewer_id) -> dict:
    me = str(viewer_id)
    row = _visible_row(note_id, me)
    return _shape([row], me)[0]


def active_note_for_profile(owner_id: str, viewer_id: str) -> dict | None:
    """PROFILE: {note_id, created_at, expires_at, kind, category, pieces[<=4], text_bg?, text?}
    or None. Same audience rules (author or follower, not blocked / hidden)."""
    owner, viewer = str(owner_id), str(viewer_id)
    if not _can_see(owner, viewer):
        return None
    rows = _live_query().eq("user_id", owner).limit(1).execute().data or []
    if not rows:
        return None
    r = rows[0]
    return {"note_id": str(r["id"]), "created_at": r["created_at"],
            "expires_at": r["expires_at"], **note_thumb(r)}


# ---------------------------------------------------------------- write

def _check_rate(me: str) -> None:
    since = _iso(_now() - timedelta(hours=1))
    n = supabase.table("fit_notes").select("id", count="exact")\
        .eq("user_id", me).gte("created_at", since).limit(1).execute().count or 0
    if n >= RATE_PER_HOUR:
        exc = _bad("RATE_LIMITED", "You're posting notes too fast. Try again later.", 429)
        exc.headers = {"Retry-After": "600"}
        raise exc


def create_note(user_id, body: dict) -> dict:
    from pipelines.comments_pipeline import require_handle
    me = str(user_id)
    require_handle(me)                                     # 403 HANDLE_REQUIRED

    text = _clean_text(body.get("text"))
    pieces, category = [], None
    if body.get("source_type") is not None:
        pieces, category = _from_source(me, body.get("source_type"), body.get("source_id"),
                                        body.get("item_ids"))
        if not pieces:
            raise _bad("OUTFIT_EMPTY", "That outfit has no pieces left to show.")
    if pieces:
        kind = "outfit_text" if text else "outfit"
        text_bg = None
    else:
        if not text:
            raise _bad("TEXT_REQUIRED", "Write something or pick an outfit.")
        kind = "text"
        text_bg = str(body.get("text_bg") or "linen").lower()
        if text_bg not in TEXT_BGS:
            raise _bad("TEXT_BG_INVALID", "text_bg must be linen, blush or ink.")

    _check_rate(me)
    try:
        res = supabase.rpc("post_fit_note", {
            "p_user": me, "p_kind": kind, "p_text": text, "p_text_bg": text_bg,
            "p_category": category,
            "p_source_type": str(body.get("source_type")).lower() if pieces else None,
            "p_source_id": str(body.get("source_id")) if pieces and body.get("source_id") else None,
            "p_pieces": pieces, "p_ttl_minutes": NOTE_TTL_MINUTES,
        }).execute()
    except APIError as exc:
        if getattr(exc, "code", None) == "23505":          # posted twice at the same moment
            raise _bad("NOTE_CONFLICT", "Your note was just updated. Try again.", 409)
        raise
    row = res.data[0] if isinstance(res.data, list) else res.data
    logger.info("fit note posted user=%s kind=%s pieces=%s", me, kind, len(pieces))
    return _shape([row], me)[0]


def delete_note(note_id, user_id) -> dict:
    me = str(user_id)
    row = _get_row(note_id)
    if not row or row.get("deleted_at"):
        raise _not_found()
    if str(row["user_id"]) != me:
        raise _not_allowed("Only the author can delete a note.")
    supabase.table("fit_notes").update({"deleted_at": _iso(_now())})\
        .eq("id", str(row["id"])).eq("user_id", me).execute()
    return {"deleted": True}


# ---------------------------------------------------------------- reactions

def react(note_id, user_id, reaction: str) -> dict:
    from pipelines.activity_pipeline import record
    me = str(user_id)
    reaction = str(reaction or "").lower()
    if reaction not in REACTIONS:
        raise _bad("INVALID_REACTION", f"Reaction must be one of: {', '.join(REACTIONS)}.")
    row = _visible_row(note_id, me)                         # 404 / 410
    owner, nid = str(row["user_id"]), str(row["id"])
    if owner == me:
        raise _bad("CANNOT_REACT_OWN", "You can't react to your own note.")
    supabase.table("fit_note_reactions").upsert({
        "note_id": nid, "user_id": me, "reaction": reaction, "created_at": _iso(_now()),
    }, on_conflict="note_id,user_id").execute()
    record(owner, me, "note_reacted", note_id=nid, reaction_type=reaction)   # + push
    return {"my_reaction": reaction}


def unreact(note_id, user_id) -> dict:
    me = str(user_id)
    row = _get_row(note_id)
    if row:
        supabase.table("fit_note_reactions").delete()\
            .eq("note_id", str(row["id"])).eq("user_id", me).execute()
    return {"my_reaction": None}


def _encode(ts: str, uid: str) -> str:
    return base64.urlsafe_b64encode(f"{ts}|{uid}".encode()).decode()


def _decode(cursor: str) -> tuple[str, str]:
    try:
        ts, uid = base64.urlsafe_b64decode(cursor.encode()).decode().split("|", 1)
        UUID(uid)
        return ts, uid
    except Exception:
        raise _bad("INVALID_CURSOR", "That page cursor isn't valid. Start from the first page.")


def list_reactions(note_id, user_id, type_: str | None = None,
                   cursor: str | None = None, limit: int = DEFAULT_PAGE) -> dict:
    """Owner only, live only."""
    from pipelines.users_pipeline import build_users, unknown_user
    me = str(user_id)
    row = _get_row(note_id)
    if not row or row.get("deleted_at") or str(row["user_id"]) != me:
        raise _not_found()
    if not is_live(row):
        raise _expired()
    if type_ is not None and type_ not in REACTIONS:
        raise _bad("INVALID_REACTION", f"Reaction must be one of: {', '.join(REACTIONS)}.")
    nid = str(row["id"])
    excluded = excluded_user_ids(me)
    limit = max(1, min(int(limit), MAX_PAGE))

    all_rows = not_in(supabase.table("fit_note_reactions").select("reaction").eq("note_id", nid),
                      "user_id", excluded).execute().data or []
    totals = {r: 0 for r in REACTIONS}
    for r in all_rows:
        if r["reaction"] in totals:
            totals[r["reaction"]] += 1
    totals["all"] = sum(totals.values())

    q = not_in(supabase.table("fit_note_reactions").select("user_id, reaction, created_at")
               .eq("note_id", nid), "user_id", excluded)
    if type_:
        q = q.eq("reaction", type_)
    if cursor:
        ts, uid = _decode(cursor)
        q = q.or_(f'created_at.lt."{ts}",and(created_at.eq."{ts}",user_id.lt.{uid})')
    rows = q.order("created_at", desc=True).order("user_id", desc=True)\
        .limit(limit + 1).execute().data or []
    has_more = len(rows) > limit
    rows = rows[:limit]
    people = build_users([r["user_id"] for r in rows], me)
    items = []
    for r in rows:
        u = people.get(str(r["user_id"])) or unknown_user(r["user_id"], me)
        items.append({**u, "user": u, "reaction": r["reaction"], "reacted_at": r["created_at"]})
    last = rows[-1] if rows else None
    return {"totals": totals, "items": items,
            "next_cursor": _encode(str(last["created_at"]), str(last["user_id"])) if has_more and last else None}


# ---------------------------------------------------------------- report

def report_note(note_id, user_id, reason, note=None) -> dict:
    from pipelines.safety_pipeline import _clean_report, _save_report
    me = str(user_id)
    reason, note = _clean_report(reason, note)
    row = _visible_row(note_id, me)                         # 404 / 410
    if str(row["user_id"]) == me:
        raise _bad("CANNOT_REPORT_SELF", "You can't report your own note.")
    _save_report(me, "note", str(row["id"]), reason, note)
    logger.info("report note=%s by=%s reason=%s", row["id"], me, reason)
    return {"reported": True}


# ---------------------------------------------------------------- activity / push helpers

def notes_by_id(note_ids: list[str]) -> dict[str, dict]:
    ids = [str(i) for i in note_ids if i]
    if not ids:
        return {}
    rows = supabase.table("fit_notes").select(NOTE_FIELDS).in_("id", ids).execute().data or []
    return {str(r["id"]): r for r in rows}