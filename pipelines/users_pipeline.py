"""
Users (Discover part 2): B9 handles (2026-10-31), B10 public profiles (2026-11-02),
B10.4 public profile photo (2026-11-03), B11 follow lists (2026-11-03).

One place that builds the public `user` object every social response uses:
    {user_id, display_name, handle, avatar_url, avatar_url_small, is_following, is_me}

avatar_url / avatar_url_small come ONLY from the public profile photo the user
uploaded on purpose (B10.4: profiles.photo_key / photo_small_key, 400 / 96 px).
Never from the try-on avatar (AV-7 / 0.2 privacy rule). No photo -> both null.

Handles (profiles.handle):
    * lowercase, 3-20 chars from a-z 0-9 _ . ; no '.' at the start or end
      (also enforced by the database check constraint)
    * unique (unique index), reserved words refused
    * null until the user picks one
    * N1 (2026-11-02): the FIRST pick never starts the lock, so a new user can
      fix a typo. The lock starts at the first CHANGE: after that, one change
      every HANDLE_CHANGE_DAYS days.

Public profile (B10): profiles.bio (<=150), city (<=40), style_tags (<=5 from
STYLE_TAGS). Counts are live: public posts, followers, following, reactions
received on public posts.
"""

import base64
import logging
import random
import re
from datetime import datetime, timedelta, timezone
from uuid import UUID

from postgrest.exceptions import APIError

from core.storage.s3 import storage
from core.storage.supabase_client import supabase
from shared.exceptions import FitCheckException

logger = logging.getLogger(__name__)

HANDLE_RE = re.compile(r"^(?!\.)[a-z0-9_.]{3,20}(?<!\.)$")
HANDLE_CHANGE_DAYS = 30
RESERVED_HANDLES = {
    "admin", "administrator", "fitcheck", "fitcheckapp", "fitcheck_app",
    "fitcheck.app", "support", "help", "official", "staff", "team",
    "moderator", "mod", "mods", "root", "system", "api", "www", "app",
    "settings", "about", "login", "signup", "signin", "logout", "me",
    "null", "undefined", "everyone", "here", "security", "privacy",
}

MAX_BIO_LEN = 150
MAX_CITY_LEN = 40
MAX_STYLE_TAGS = 5
STYLE_TAGS = ("casual", "formal", "streetwear", "business", "athletic", "vintage")
# letters (incl. accents/tone marks, e.g. Ile-Ife with Yoruba marks), spaces . ' -
CITY_RE = re.compile(r"^[^\W\d_](?:[^\W\d_]|[\u0300-\u036f]|[ .'\-])*$", re.UNICODE)
DEFAULT_PAGE = 20
MAX_PAGE = 50
SEARCH_MATCH_CAP = 300      # B11 ?q=: at most this many name/handle matches are checked


# ---------------------------------------------------------------- errors

def _bad(code: str, message: str, status: int = 400) -> FitCheckException:
    return FitCheckException(message, code=code, status_code=status)


def _user_not_found() -> FitCheckException:
    return _bad("USER_NOT_FOUND", "User not found.", 404)


# ---------------------------------------------------------------- handles

def normalize_handle(raw) -> str:
    """'  @Ada.Okafor ' -> 'ada.okafor'"""
    return str(raw or "").strip().lstrip("@").strip().lower()


def handle_problem(handle: str) -> str | None:
    """None if the handle is allowed, otherwise the error code."""
    if not HANDLE_RE.match(handle):
        return "HANDLE_INVALID"
    if handle in RESERVED_HANDLES:
        return "HANDLE_RESERVED"
    return None


def _owner_of(handle: str) -> str | None:
    res = supabase.table("profiles").select("id")\
        .eq("handle", handle).limit(1).execute()
    return str(res.data[0]["id"]) if res.data else None


def _suggest(base: str, uid: str | None = None) -> str | None:
    """A free handle close to `base` (letters/digits/_ kept, digits appended)."""
    stem = re.sub(r"[^a-z0-9_]", "", base.replace(" ", "_").replace("-", "_").lower())
    stem = stem.strip("_")[:15] or "fitchecker"
    if len(stem) < 3:
        stem = (stem + "fit")[:15]
    candidates = [stem] + [f"{stem}{random.randint(10, 9999)}" for _ in range(8)]
    for c in candidates:
        if handle_problem(c) is None and _owner_of(c) in (None, uid):
            return c
    return None


def check_available(raw, user_id: UUID) -> dict:
    """GET /users/handle-available. Never errors - always says why."""
    uid = str(user_id)
    handle = normalize_handle(raw)
    problem = handle_problem(handle)
    if problem:
        return {"handle": handle, "available": False,
                "reason": "invalid" if problem == "HANDLE_INVALID" else "reserved",
                "suggestion": _suggest(handle, uid)}
    owner = _owner_of(handle)
    if owner and owner != uid:
        return {"handle": handle, "available": False, "reason": "taken",
                "suggestion": _suggest(handle, uid)}
    return {"handle": handle, "available": True, "reason": None, "suggestion": None}


# ---------------------------------------------------------------- user objects

def _following_set(viewer_id: str, user_ids: list[str]) -> set[str]:
    if not user_ids:
        return set()
    res = supabase.table("follows").select("following_id")\
        .eq("follower_id", viewer_id).in_("following_id", user_ids).execute()
    return {str(r["following_id"]) for r in (res.data or [])}


def build_users(user_ids: list[str], viewer_id: str) -> dict[str, dict]:
    """user_id -> public user object, for many users in 2 queries."""
    ids = list({str(u) for u in user_ids if u})
    if not ids:
        return {}
    rows = supabase.table("profiles")\
        .select("id, full_name, handle, photo_key, photo_small_key")\
        .in_("id", ids).execute().data or []
    following = _following_set(viewer_id, ids)
    users = {}
    for r in rows:
        uid = str(r["id"])
        users[uid] = {
            "user_id": uid,
            "display_name": r.get("full_name"),
            "handle": r.get("handle"),
            # B10.4: ONLY the public profile photo - see module docstring.
            "avatar_url": storage.get_url(r["photo_key"]) if r.get("photo_key") else None,
            "avatar_url_small": storage.get_url(r["photo_small_key"]) if r.get("photo_small_key") else None,
            "is_following": uid in following,
            "is_me": uid == viewer_id,
        }
    return users


def unknown_user(user_id: str, viewer_id: str) -> dict:
    """Shape for a user whose profile row is missing (deleted account)."""
    return {"user_id": str(user_id), "display_name": None, "handle": None,
            "avatar_url": None, "avatar_url_small": None,
            "is_following": False, "is_me": str(user_id) == viewer_id}


# ---------------------------------------------------------------- me / lookup

def _me_row(uid: str) -> dict:
    res = supabase.table("profiles").select("id, handle, handle_changed_at")\
        .eq("id", uid).limit(1).execute()
    if not res.data:
        raise _user_not_found()
    return res.data[0]


def _can_change_at(row: dict) -> str | None:
    """When the next handle change is allowed (None = allowed now)."""
    if not row.get("handle") or not row.get("handle_changed_at"):
        return None
    changed = datetime.fromisoformat(str(row["handle_changed_at"]).replace("Z", "+00:00"))
    nxt = changed + timedelta(days=HANDLE_CHANGE_DAYS)
    return nxt.isoformat() if nxt > datetime.now(timezone.utc) else None


def get_me(user_id: UUID) -> dict:
    uid = str(user_id)
    row = _me_row(uid)
    me = build_users([uid], uid).get(uid) or unknown_user(uid, uid)
    me["handle_can_change_at"] = _can_change_at(row)
    return me


def set_handle(user_id: UUID, raw) -> dict:
    uid = str(user_id)
    handle = normalize_handle(raw)
    problem = handle_problem(handle)
    if problem == "HANDLE_INVALID":
        raise _bad("HANDLE_INVALID",
                   "Handles are 3-20 characters: a-z, 0-9, _ and . (not at the start or end).")
    if problem == "HANDLE_RESERVED":
        raise _bad("HANDLE_RESERVED", "That handle is reserved. Try another one.")

    row = _me_row(uid)
    if row.get("handle") == handle:
        return get_me(user_id)                      # no change - nothing to do

    owner = _owner_of(handle)
    if owner and owner != uid:
        raise _bad("HANDLE_TAKEN", "That handle is taken.", 409)

    next_at = _can_change_at(row)
    if next_at:
        raise _bad("HANDLE_CHANGE_TOO_SOON",
                   f"You can change your handle once every {HANDLE_CHANGE_DAYS} days. "
                   f"Next change: {next_at[:10]}.", 429)

    # N1: picking a handle for the first time doesn't start the lock.
    first_pick = not row.get("handle")
    try:
        supabase.table("profiles").update({
            "handle": handle,
            "handle_changed_at": None if first_pick else datetime.now(timezone.utc).isoformat(),
        }).eq("id", uid).execute()
    except APIError as exc:
        if getattr(exc, "code", None) == "23505":   # taken by a parallel request
            raise _bad("HANDLE_TAKEN", "That handle is taken.", 409)
        raise
    logger.info("handle set user=%s handle=%s", uid, handle)
    return get_me(user_id)


def get_by_handle(raw, viewer_id: UUID) -> dict:
    handle = normalize_handle(raw)
    owner = _owner_of(handle) if handle else None
    if not owner:
        raise _user_not_found()
    viewer = str(viewer_id)
    return build_users([owner], viewer).get(owner) or unknown_user(owner, viewer)


# ---------------------------------------------------------------- B10 public profile

def _profile_row(uid: str) -> dict | None:
    res = supabase.table("profiles")\
        .select("id, bio, city, style_tags, created_at")\
        .eq("id", uid).limit(1).execute()
    return res.data[0] if res.data else None


def _count(table: str, column: str, **filters) -> int:
    q = supabase.table(table).select(column, count="exact")
    for k, v in filters.items():
        q = q.eq(k, v)
    return q.limit(1).execute().count or 0


def _reactions_received(uid: str) -> int:
    rows = supabase.table("public_post_rank").select("total_reactions")\
        .eq("user_id", uid).execute().data or []
    return sum(int(r.get("total_reactions") or 0) for r in rows)


def _follows_me(owner: str, viewer: str) -> bool:
    if owner == viewer:
        return False
    res = supabase.table("follows").select("follower_id")\
        .eq("follower_id", owner).eq("following_id", viewer).limit(1).execute()
    return bool(res.data)


def public_profile(target_id, viewer_id) -> dict:
    """B10.1 - 404 USER_NOT_FOUND if the user doesn't exist.
    (E18 will also return 404 when either side has blocked the other.)"""
    uid, viewer = str(target_id), str(viewer_id)
    row = _profile_row(uid)
    if not row:
        raise _user_not_found()
    user = build_users([uid], viewer).get(uid) or unknown_user(uid, viewer)
    return {
        "user": user,
        "bio": row.get("bio"),
        "city": row.get("city"),
        "style_tags": row.get("style_tags") or [],
        "counts": {
            "posts": _count("outfits", "outfit_id", user_id=uid, is_public=True),
            "followers": _count("follows", "follower_id", following_id=uid),
            "following": _count("follows", "following_id", follower_id=uid),
            "reactions_received": _reactions_received(uid),
        },
        "follows_me": _follows_me(uid, viewer),
        "joined_at": row.get("created_at"),
    }


def public_profile_by_handle(raw, viewer_id) -> dict:
    handle = normalize_handle(raw)
    owner = _owner_of(handle) if handle else None
    if not owner:
        raise _user_not_found()
    return public_profile(owner, viewer_id)


def update_public_profile(user_id, sent: dict) -> dict:
    """B10.3 - only the fields present in `sent` change. null / "" clears."""
    uid = str(user_id)
    if not _profile_row(uid):
        raise _user_not_found()
    updates = {}

    if "bio" in sent:
        bio = str(sent["bio"] or "").strip()
        if len(bio) > MAX_BIO_LEN:
            raise _bad("BIO_TOO_LONG", f"Bio must be {MAX_BIO_LEN} characters or fewer.")
        updates["bio"] = bio or None

    if "city" in sent:
        city = " ".join(str(sent["city"] or "").split())
        if city and (len(city) > MAX_CITY_LEN or not CITY_RE.match(city)):
            raise _bad("CITY_INVALID",
                       f"City must be up to {MAX_CITY_LEN} letters (spaces, . ' - allowed).")
        updates["city"] = city or None

    if "style_tags" in sent:
        tags = []
        for t in (sent["style_tags"] or []):
            tag = str(t or "").strip().lower()
            if tag not in STYLE_TAGS:
                raise _bad("STYLE_TAG_INVALID",
                           f"Style tags must be from: {', '.join(STYLE_TAGS)}.")
            if tag not in tags:
                tags.append(tag)
        if len(tags) > MAX_STYLE_TAGS:
            raise _bad("STYLE_TAG_INVALID", f"Pick up to {MAX_STYLE_TAGS} style tags.")
        updates["style_tags"] = tags

    if updates:
        supabase.table("profiles").update(updates).eq("id", uid).execute()
    return public_profile(uid, uid)


# ---------------------------------------------------------------- B10.2 posts, cursor paging

def _encode_cursor(ts: str, outfit_id: str) -> str:
    return base64.urlsafe_b64encode(f"{ts}|{outfit_id}".encode()).decode()


def _decode_cursor(cursor: str) -> tuple[str, str]:
    try:
        ts, oid = base64.urlsafe_b64decode(cursor.encode()).decode().split("|", 1)
        UUID(oid)
        return ts, oid
    except Exception:
        raise _bad("INVALID_CURSOR", "That page cursor isn't valid. Start from the first page.")


def user_posts(target_id, viewer_id, cursor: str | None = None,
               limit: int = DEFAULT_PAGE, include_private: bool = False) -> dict:
    """
    B10.2 - newest first, keyset paging on (time, outfit_id).
    Public posts are ordered by published_at. include_private=true only works
    on your own profile: then ALL your outfits are returned, ordered by
    created_at, with is_public false on the unpublished ones.
    """
    # Imported here to avoid a circular import.
    from api.social import FEED_FIELDS, _decorate_posts

    uid, viewer = str(target_id), str(viewer_id)
    if not _profile_row(uid):
        raise _user_not_found()
    limit = max(1, min(int(limit), MAX_PAGE))
    private = include_private and uid == viewer
    col = "created_at" if private else "published_at"

    q = supabase.table("outfits").select(FEED_FIELDS).eq("user_id", uid)
    if not private:
        q = q.eq("is_public", True)
    if cursor:
        ts, oid = _decode_cursor(cursor)
        q = q.or_(f'{col}.lt."{ts}",and({col}.eq."{ts}",outfit_id.lt.{oid})')
    rows = q.order(col, desc=True).order("outfit_id", desc=True)\
        .limit(limit + 1).execute().data or []

    has_more = len(rows) > limit
    rows = rows[:limit]
    _decorate_posts(rows, viewer)
    next_cursor = None
    if has_more and rows:
        last = rows[-1]
        next_cursor = _encode_cursor(str(last[col]), str(last["outfit_id"]))
    return {"items": rows, "next_cursor": next_cursor}


# ---------------------------------------------------------------- B11 follow lists

def _search_term(raw) -> str:
    """'@Ada O' -> 'ada o'. Only letters, digits, space and _ . ' - are kept
    (anything else could break the PostgREST filter)."""
    term = str(raw or "").strip().lstrip("@").strip().lower()
    term = re.sub(r"[^\w .'\-]", "", term, flags=re.UNICODE)
    return " ".join(term.split())[:40]


def _matching_profile_ids(term: str) -> list[str]:
    """Profiles whose handle starts with the term, or whose name (any word) starts with it."""
    res = supabase.table("profiles").select("id")\
        .or_(f'handle.ilike."{term}*",full_name.ilike."{term}*",full_name.ilike."* {term}*"')\
        .limit(SEARCH_MATCH_CAP).execute()
    return [str(r["id"]) for r in (res.data or [])]


def follow_list(target_id, viewer_id, direction: str, cursor: str | None = None,
                limit: int = DEFAULT_PAGE, q: str | None = None) -> dict:
    """
    B11 - direction 'followers' (people who follow target) or 'following'
    (people target follows). Newest follow first, keyset paging on
    (follows.created_at, person id). is_following is from the VIEWER's side.
    """
    uid, viewer = str(target_id), str(viewer_id)
    if not _profile_row(uid):
        raise _user_not_found()
    limit = max(1, min(int(limit), MAX_PAGE))
    if direction == "followers":
        match_col, person_col = "following_id", "follower_id"
    else:
        match_col, person_col = "follower_id", "following_id"

    query = supabase.table("follows").select(f"{person_col}, created_at").eq(match_col, uid)

    term = _search_term(q) if q else ""
    if term:
        ids = _matching_profile_ids(term)
        if not ids:
            return {"items": [], "next_cursor": None}
        query = query.in_(person_col, ids)

    if cursor:
        ts, pid = _decode_cursor(cursor)
        query = query.or_(f'created_at.lt."{ts}",and(created_at.eq."{ts}",{person_col}.lt.{pid})')

    rows = query.order("created_at", desc=True).order(person_col, desc=True)\
        .limit(limit + 1).execute().data or []
    has_more = len(rows) > limit
    rows = rows[:limit]

    people = build_users([r[person_col] for r in rows], viewer)
    items = [people.get(str(r[person_col])) or unknown_user(r[person_col], viewer) for r in rows]

    next_cursor = None
    if has_more and rows:
        last = rows[-1]
        next_cursor = _encode_cursor(str(last["created_at"]), str(last[person_col]))
    return {"items": items, "next_cursor": next_cursor}


def remove_follower(user_id, follower_id) -> dict:
    """B11 - stop someone following me. Idempotent (200 even if they didn't)."""
    supabase.table("follows").delete()\
        .eq("follower_id", str(follower_id)).eq("following_id", str(user_id)).execute()
    logger.info("follower removed user=%s follower=%s", user_id, follower_id)
    return {"removed": True, "user_id": str(follower_id)}