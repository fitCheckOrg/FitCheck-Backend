"""
Collections (COL-1) - named folders of saved looks, saved outfits
and (A7, 2026-10-31) saved Discover posts. Private by default; COLLECTIONS-PUBLIC-1
(2026-11-06) lets the owner make one public.

Optional on top of saving: a look/outfit is saved first, then (optionally)
filed into one or more collections.

Tables (see sql/collections.sql):
    collections         (collection_id, user_id, name, is_public, created_at, updated_at)
                        unique (user_id, lower(btrim(name)))
    collection_entries  (entry_id, collection_id, user_id, look_id | outfit_id | post_id, added_at)
                        exactly one of the three. look_id / outfit_id: FK ON DELETE CASCADE.
                        post_id: NO foreign key on purpose - a post entry is kept when the
                        post is unpublished or deleted and shown as {unavailable: true}.

Rules:
    * one look/outfit can be in several collections
    * deleting a collection never deletes its looks/outfits
    * deleting a look/outfit removes it from every collection (DB cascade)
    * a post entry stays when the post goes away -> {"unavailable": true}
    * adding a post to a collection also bookmarks it (saved_outfits)
    * name 1-40 chars after trimming, unique per user (case-insensitive)
    * every query is scoped to the user - another user's ids give 404
      (except the public views below)

COLLECTIONS-PUBLIC-1 - public view (anyone but the owner):
    * only collections with is_public = true; private ones 404 COLLECTION_NOT_FOUND
    * only POST entries are ever shown. Looks (try-on photos of the owner) and
      private outfits are NEVER shown to anyone else, even in a public collection.
    * posts that are unpublished, under review, or by people blocked either way /
      hidden for the viewer are skipped and not counted in post_count
    * owner and viewer blocked either way -> 404 (USER_NOT_FOUND / COLLECTION_NOT_FOUND)
"""

import base64
import logging
from datetime import datetime, timezone
from uuid import UUID

from postgrest.exceptions import APIError

from core.storage.supabase_client import supabase
from shared.exceptions import FitCheckException
from pipelines.tryon_looks_pipeline import _to_look
from pipelines.safety_pipeline import excluded_user_ids, is_blocked_either_way

logger = logging.getLogger(__name__)

COLLECTIONS = "collections"
ENTRIES = "collection_entries"
MAX_NAME_LEN = 40
MAX_COLLECTIONS = 100
MAX_PAGE = 100
COVER_POSTS = 4
COVER_PIECES = 4
PUBLIC_PAGE = 20
ENTRY_TYPES = {"look": "look_id", "outfit": "outfit_id", "post": "post_id"}


# ---------------------------------------------------------------- errors

def _not_found(code: str, message: str) -> FitCheckException:
    return FitCheckException(message, code=code, status_code=404)


def _collection_not_found() -> FitCheckException:
    return _not_found("COLLECTION_NOT_FOUND", "Collection not found.")


def _name_taken() -> FitCheckException:
    return FitCheckException(
        "You already have a collection with that name.",
        code="COLLECTION_NAME_TAKEN", status_code=409,
    )


def _is_unique_violation(exc: APIError) -> bool:
    return getattr(exc, "code", None) == "23505"


# ---------------------------------------------------------------- helpers

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_name(name) -> str:
    clean = " ".join(str(name or "").split())   # trim + collapse inner spaces
    if not clean or len(clean) > MAX_NAME_LEN:
        raise FitCheckException(
            f"Collection name must be 1-{MAX_NAME_LEN} characters.",
            code="COLLECTION_NAME_INVALID", status_code=400,
        )
    return clean


def _entry_column(entry_type: str) -> str:
    col = ENTRY_TYPES.get(str(entry_type or "").lower())
    if not col:
        raise FitCheckException(
            "entry_type must be 'look', 'outfit' or 'post'.",
            code="INVALID_ENTRY_TYPE", status_code=400,
        )
    return col


def _get_collection_row(uid: str, collection_id: str) -> dict:
    res = supabase.table(COLLECTIONS).select("*")\
        .eq("collection_id", collection_id).eq("user_id", uid)\
        .limit(1).execute()
    if not res.data:
        raise _collection_not_found()
    return res.data[0]


def _name_in_use(uid: str, name: str, except_id: str | None = None) -> bool:
    # A user has at most MAX_COLLECTIONS rows, so compare in Python (case-insensitive).
    res = supabase.table(COLLECTIONS).select("collection_id, name")\
        .eq("user_id", uid).execute()
    for row in (res.data or []):
        if row["name"].strip().lower() == name.lower() and row["collection_id"] != except_id:
            return True
    return False


def _touch(collection_id: str) -> None:
    supabase.table(COLLECTIONS).update({"updated_at": _now()})\
        .eq("collection_id", collection_id).execute()


def _summaries(uid: str, rows: list[dict]) -> list[dict]:
    """Add item_count + cover_image_url to collection rows (2-3 queries total)."""
    if not rows:
        return []
    ids = [r["collection_id"] for r in rows]

    entries = supabase.table(ENTRIES).select("collection_id, look_id, outfit_id, added_at")\
        .eq("user_id", uid).in_("collection_id", ids)\
        .order("added_at", desc=True).execute().data or []

    counts = {cid: 0 for cid in ids}
    newest_look = {}                      # collection_id -> look_id (newest added)
    for e in entries:
        cid = e["collection_id"]
        counts[cid] = counts.get(cid, 0) + 1
        if e.get("look_id") and cid not in newest_look:
            newest_look[cid] = e["look_id"]

    cover_by_look = {}
    if newest_look:
        looks = supabase.table("tryon_looks").select("*")\
            .eq("user_id", uid).in_("look_id", list(set(newest_look.values())))\
            .execute().data or []
        cover_by_look = {l["look_id"]: _to_look(l)["image_url"] for l in looks}

    out = []
    for r in rows:
        cid = r["collection_id"]
        out.append({
            "collection_id": cid,
            "name": r["name"],
            "item_count": counts.get(cid, 0),
            "cover_image_url": cover_by_look.get(newest_look.get(cid)),
            "is_public": bool(r.get("is_public")),          # COLLECTIONS-PUBLIC-1
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        })
    return out


def _post_visible(uid: str, post_id: str) -> bool:
    """Only a published (public) post can be filed as a post - yours or anyone's.
    Your own unpublished outfit goes into a collection as entry_type "outfit"."""
    res = supabase.table("outfits").select("outfit_id, is_public")\
        .eq("outfit_id", post_id).limit(1).execute()
    row = res.data[0] if res.data else None
    return bool(row and row.get("is_public"))


def _bookmark_post(uid: str, post_id: str) -> None:
    """A7: filing a post into a collection also turns its social bookmark on."""
    existing = supabase.table("saved_outfits").select("outfit_id")\
        .eq("user_id", uid).eq("outfit_id", post_id).limit(1).execute()
    if existing.data:
        return
    try:
        supabase.table("saved_outfits").insert({"user_id": uid, "outfit_id": post_id}).execute()
    except APIError as exc:
        if not _is_unique_violation(exc):
            raise


def _check_target_owned(uid: str, entry_type: str, entry_id: str) -> None:
    if entry_type == "post":
        if not _post_visible(uid, entry_id):
            raise _not_found("POST_NOT_FOUND", "This post isn't available.")
        return
    if entry_type == "look":
        res = supabase.table("tryon_looks").select("look_id")\
            .eq("look_id", entry_id).eq("user_id", uid).limit(1).execute()
        if not res.data:
            raise _not_found("LOOK_NOT_FOUND", "Look not found.")
    else:
        res = supabase.table("outfits").select("outfit_id")\
            .eq("outfit_id", entry_id).eq("user_id", uid).limit(1).execute()
        if not res.data:
            raise _not_found("OUTFIT_NOT_FOUND", "Outfit not found.")


# ---------------------------------------------------------------- collections

def list_collections(user_id: UUID, contains_type: str | None = None,
                     contains_id: UUID | None = None) -> dict:
    uid = str(user_id)

    want_contains = contains_type is not None or contains_id is not None
    if want_contains and (contains_type is None or contains_id is None):
        raise FitCheckException(
            "Send both contains_type and contains_id, or neither.",
            code="INVALID_QUERY", status_code=400,
        )

    rows = supabase.table(COLLECTIONS).select("*")\
        .eq("user_id", uid).order("updated_at", desc=True)\
        .execute().data or []
    collections = _summaries(uid, rows)

    if want_contains:
        col = _entry_column(contains_type)
        hits = supabase.table(ENTRIES).select("collection_id")\
            .eq("user_id", uid).eq(col, str(contains_id)).execute().data or []
        inside = {h["collection_id"] for h in hits}
        for c in collections:
            c["contains"] = c["collection_id"] in inside

    return {"collections": collections}


def create_collection(user_id: UUID, name) -> dict:
    uid = str(user_id)
    clean = _clean_name(name)

    count = supabase.table(COLLECTIONS).select("collection_id", count="exact")\
        .eq("user_id", uid).execute().count or 0
    if count >= MAX_COLLECTIONS:
        raise FitCheckException(
            f"You can have up to {MAX_COLLECTIONS} collections.",
            code="COLLECTION_LIMIT_REACHED", status_code=400,
        )
    if _name_in_use(uid, clean):
        raise _name_taken()

    try:
        res = supabase.table(COLLECTIONS).insert({"user_id": uid, "name": clean}).execute()
    except APIError as exc:
        if _is_unique_violation(exc):        # created by a parallel request
            raise _name_taken()
        raise
    logger.info("collection created user=%s name=%s", uid, clean)
    return _summaries(uid, res.data)[0]


def rename_collection(user_id: UUID, collection_id: UUID, name) -> dict:
    uid, cid = str(user_id), str(collection_id)
    clean = _clean_name(name)
    _get_collection_row(uid, cid)
    if _name_in_use(uid, clean, except_id=cid):
        raise _name_taken()

    try:
        res = supabase.table(COLLECTIONS).update({"name": clean, "updated_at": _now()})\
            .eq("collection_id", cid).eq("user_id", uid).execute()
    except APIError as exc:
        if _is_unique_violation(exc):
            raise _name_taken()
        raise
    if not res.data:
        raise _collection_not_found()
    return _summaries(uid, res.data)[0]


def update_collection(user_id: UUID, collection_id: UUID, sent: dict) -> dict:
    """PATCH: only the fields sent change. name -> rename (same rules as before);
    is_public (COLLECTIONS-PUBLIC-1) -> show / hide it on your profile."""
    uid, cid = str(user_id), str(collection_id)
    _get_collection_row(uid, cid)
    if sent.get("name") is not None:
        rename_collection(user_id, collection_id, sent["name"])
    if "is_public" in sent and sent["is_public"] is not None:
        supabase.table(COLLECTIONS).update({"is_public": bool(sent["is_public"])})\
            .eq("collection_id", cid).eq("user_id", uid).execute()
        logger.info("collection visibility user=%s collection=%s public=%s", uid, cid, sent["is_public"])
    return _summaries(uid, [_get_collection_row(uid, cid)])[0]


def delete_collection(user_id: UUID, collection_id: UUID) -> dict:
    """Deletes the folder only; its entries cascade, looks/outfits stay."""
    res = supabase.table(COLLECTIONS).delete()\
        .eq("collection_id", str(collection_id)).eq("user_id", str(user_id))\
        .execute()
    if not res.data:
        raise _collection_not_found()
    return {"collection_id": str(collection_id), "deleted": True}


# ---------------------------------------------------------------- entries

def list_entries(user_id: UUID, collection_id: UUID,
                 limit: int = 50, offset: int = 0) -> dict:
    uid, cid = str(user_id), str(collection_id)
    _get_collection_row(uid, cid)
    limit = max(1, min(int(limit), MAX_PAGE))
    offset = max(0, int(offset))

    entries = supabase.table(ENTRIES).select("look_id, outfit_id, post_id, added_at")\
        .eq("collection_id", cid).eq("user_id", uid)\
        .order("added_at", desc=True)\
        .range(offset, offset + limit - 1)\
        .execute().data or []

    look_ids = [e["look_id"] for e in entries if e.get("look_id")]
    outfit_ids = [e["outfit_id"] for e in entries if e.get("outfit_id")]
    post_ids = [e["post_id"] for e in entries if e.get("post_id")]

    looks = {}
    if look_ids:
        rows = supabase.table("tryon_looks").select("*")\
            .eq("user_id", uid).in_("look_id", look_ids).execute().data or []
        looks = {r["look_id"]: _to_look(r) for r in rows}

    outfits = {}
    if outfit_ids:
        rows = supabase.table("outfits").select("*")\
            .eq("user_id", uid).in_("outfit_id", outfit_ids).execute().data or []
        outfits = {r["outfit_id"]: r for r in rows}

    posts = _load_posts(post_ids, uid) if post_ids else {}

    out = []
    for e in entries:
        if e.get("look_id") and e["look_id"] in looks:
            out.append({"entry_type": "look", "added_at": e["added_at"], "look": looks[e["look_id"]]})
        elif e.get("outfit_id") and e["outfit_id"] in outfits:
            out.append({"entry_type": "outfit", "added_at": e["added_at"], "outfit": outfits[e["outfit_id"]]})
        elif e.get("post_id"):
            pid = e["post_id"]
            if pid in posts:
                out.append({"entry_type": "post", "entry_id": pid,
                            "added_at": e["added_at"], "post": posts[pid]})
            else:
                # Unpublished, deleted, or no longer visible: keep the entry,
                # send no post data. The app shows "No longer available".
                out.append({"entry_type": "post", "entry_id": pid,
                            "added_at": e["added_at"], "unavailable": True})

    return {"entries": out, "limit": limit, "offset": offset}


def _load_posts(post_ids: list[str], uid: str) -> dict[str, dict]:
    """Published posts in the Discover feed shape. A post that isn't public any
    more (unpublished or deleted) is left out -> the entry shows as unavailable,
    for everyone including its owner (their outfit is still in their Lookbook)."""
    # Imported here to avoid a circular import (api.social is the post owner).
    from api.social import FEED_FIELDS, _decorate_posts

    rows = supabase.table("outfits").select(FEED_FIELDS)\
        .in_("outfit_id", post_ids).execute().data or []
    rows = [r for r in rows if r.get("is_public")]
    _decorate_posts(rows, uid)
    return {r["outfit_id"]: r for r in rows}


def add_entry(user_id: UUID, collection_id: UUID, entry_type: str,
              entry_id: UUID) -> tuple[dict, bool]:
    """Returns (entry, created). created=False when it was already in the collection."""
    uid, cid, eid = str(user_id), str(collection_id), str(entry_id)
    col = _entry_column(entry_type)
    etype = entry_type.lower()
    _get_collection_row(uid, cid)
    _check_target_owned(uid, etype, eid)

    def _existing():
        res = supabase.table(ENTRIES).select("added_at")\
            .eq("collection_id", cid).eq(col, eid).limit(1).execute()
        return res.data[0] if res.data else None

    if etype == "post":
        _bookmark_post(uid, eid)

    found = _existing()
    if found:
        return {"collection_id": cid, "entry_type": etype, "entry_id": eid,
                "added_at": found["added_at"]}, False

    try:
        res = supabase.table(ENTRIES).insert({
            "collection_id": cid, "user_id": uid, col: eid,
        }).execute()
    except APIError as exc:
        if _is_unique_violation(exc):          # added by a parallel request
            found = _existing()
            return {"collection_id": cid, "entry_type": etype, "entry_id": eid,
                    "added_at": found["added_at"] if found else _now()}, False
        raise

    _touch(cid)
    logger.info("collection entry added user=%s collection=%s %s=%s", uid, cid, etype, eid)
    return {"collection_id": cid, "entry_type": etype, "entry_id": eid,
            "added_at": res.data[0]["added_at"]}, True


def remove_entry(user_id: UUID, collection_id: UUID, entry_type: str,
                 entry_id: UUID) -> dict:
    uid, cid, eid = str(user_id), str(collection_id), str(entry_id)
    col = _entry_column(entry_type)
    _get_collection_row(uid, cid)

    res = supabase.table(ENTRIES).delete()\
        .eq("collection_id", cid).eq("user_id", uid).eq(col, eid)\
        .execute()
    if not res.data:
        raise _not_found("COLLECTION_ENTRY_NOT_FOUND", "That item isn't in this collection.")

    _touch(cid)
    return {"collection_id": cid, "entry_type": entry_type.lower(),
            "entry_id": eid, "deleted": True}


# ---------------------------------------------------------------- COLLECTIONS-PUBLIC-1

def _visible_posts(post_ids: list[str], viewer: str) -> dict[str, dict]:
    """post_id -> {outfit_id, user_id, item_ids} for posts this viewer may see:
    public, not under review (unless theirs), creator not blocked / hidden."""
    if not post_ids:
        return {}
    excluded = excluded_user_ids(viewer)
    rows = supabase.table("outfits").select("outfit_id, user_id, item_ids, is_public, under_review")\
        .in_("outfit_id", list(set(post_ids))).execute().data or []
    out = {}
    for r in rows:
        creator = str(r["user_id"])
        if not r.get("is_public") or creator in excluded:
            continue
        if r.get("under_review") and creator != viewer:
            continue
        out[str(r["outfit_id"])] = r
    return out


def user_collections(owner_id, viewer_id) -> dict:
    """GET /api/users/{user_id}/collections. Others see only public collections;
    the owner sees all of theirs (with is_public). Each collection:
    {collection_id, name, is_public, post_count, cover_posts[<=4 {post_id, items[]}],
     updated_at} (+ item_count for the owner: every entry, looks/outfits included)."""
    from api.social import _get_bulk_item_details        # avoid a circular import

    owner, viewer = str(owner_id), str(viewer_id)
    mine = owner == viewer
    prof = supabase.table("profiles").select("id").eq("id", owner).limit(1).execute().data
    if not prof or (not mine and is_blocked_either_way(owner, viewer)):
        raise _not_found("USER_NOT_FOUND", "User not found.")

    q = supabase.table(COLLECTIONS).select("*").eq("user_id", owner)
    if not mine:
        q = q.eq("is_public", True)
    rows = q.order("updated_at", desc=True).limit(MAX_PAGE).execute().data or []
    if not rows:
        return {"items": [], "next_cursor": None}

    ids = [r["collection_id"] for r in rows]
    entries = supabase.table(ENTRIES).select("collection_id, post_id, added_at")\
        .in_("collection_id", ids).order("added_at", desc=True).execute().data or []
    visible = _visible_posts([e["post_id"] for e in entries if e.get("post_id")], viewer)
    summaries = {c["collection_id"]: c for c in _summaries(owner, rows)} if mine else {}

    per = {cid: [] for cid in ids}
    for e in entries:
        pid = str(e["post_id"]) if e.get("post_id") else None
        if pid and pid in visible:
            per[e["collection_id"]].append(pid)

    cover_ids = {p for cid in ids for p in per[cid][:COVER_POSTS]}
    piece_ids = list({str(i) for p in cover_ids for i in (visible[p].get("item_ids") or [])[:COVER_PIECES]})
    pieces = _get_bulk_item_details(piece_ids)

    items = []
    for r in rows:
        cid = r["collection_id"]
        item = {
            "collection_id": cid,
            "name": r["name"],
            "is_public": bool(r.get("is_public")),
            "post_count": len(per[cid]),
            "cover_posts": [{
                "post_id": p,
                "items": [pieces.get(str(i), {"item_id": str(i), "deleted": True})
                          for i in (visible[p].get("item_ids") or [])[:COVER_PIECES]],
            } for p in per[cid][:COVER_POSTS]],
            "updated_at": r["updated_at"],
        }
        if mine:
            item["item_count"] = summaries.get(cid, {}).get("item_count", len(per[cid]))
        items.append(item)
    return {"items": items, "next_cursor": None}


def _encode(added_at: str, post_id: str) -> str:
    return base64.urlsafe_b64encode(f"{added_at}|{post_id}".encode()).decode()


def _decode(cursor: str) -> tuple[str, str]:
    try:
        ts, pid = base64.urlsafe_b64decode(cursor.encode()).decode().split("|", 1)
        UUID(pid)
        return ts, pid
    except Exception:
        raise FitCheckException("That page cursor isn't valid. Start from the first page.",
                                code="INVALID_CURSOR", status_code=400)


def collection_posts(collection_id, viewer_id, cursor: str | None = None,
                     limit: int = PUBLIC_PAGE) -> dict:
    """GET /api/collections/{id}/posts - the collection's POSTS in the feed shape,
    newest added first. Anyone if public, else only the owner (404 otherwise)."""
    from api.social import FEED_FIELDS, _decorate_posts   # avoid a circular import

    cid, viewer = str(collection_id), str(viewer_id)
    limit = max(1, min(int(limit), 50))
    res = supabase.table(COLLECTIONS).select("collection_id, user_id, name, is_public")\
        .eq("collection_id", cid).limit(1).execute().data
    col = res[0] if res else None
    if not col:
        raise _collection_not_found()
    owner = str(col["user_id"])
    if owner != viewer and (not col.get("is_public") or is_blocked_either_way(owner, viewer)):
        raise _collection_not_found()

    q = supabase.table(ENTRIES).select("post_id, added_at")\
        .eq("collection_id", cid).filter("post_id", "not.is", "null")
    if cursor:
        ts, pid = _decode(cursor)
        q = q.or_(f'added_at.lt."{ts}",and(added_at.eq."{ts}",post_id.lt.{pid})')
    rows = q.order("added_at", desc=True).order("post_id", desc=True)\
        .limit(limit + 1).execute().data or []
    has_more = len(rows) > limit
    rows = rows[:limit]

    visible = _visible_posts([r["post_id"] for r in rows], viewer)
    posts = []
    if visible:
        full = supabase.table("outfits").select(FEED_FIELDS)\
            .in_("outfit_id", list(visible)).execute().data or []
        by_id = {str(p["outfit_id"]): p for p in full}
        posts = [by_id[str(r["post_id"])] for r in rows if str(r["post_id"]) in by_id]
        _decorate_posts(posts, viewer)

    last = rows[-1] if rows else None
    return {
        "collection": {"collection_id": cid, "name": col["name"],
                       "is_public": bool(col.get("is_public")), "owner_id": owner},
        "items": posts,
        "next_cursor": _encode(str(last["added_at"]), str(last["post_id"])) if has_more and last else None,
    }