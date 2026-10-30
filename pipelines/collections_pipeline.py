"""
Collections (COL-1) - named, private folders of saved looks and saved outfits.

Optional on top of saving: a look/outfit is saved first, then (optionally)
filed into one or more collections.

Tables (see sql/collections.sql):
    collections         (collection_id, user_id, name, created_at, updated_at)
                        unique (user_id, lower(btrim(name)))
    collection_entries  (entry_id, collection_id, user_id, look_id | outfit_id, added_at)
                        exactly one of look_id / outfit_id; FKs ON DELETE CASCADE

Rules:
    * one look/outfit can be in several collections
    * deleting a collection never deletes its looks/outfits
    * deleting a look/outfit removes it from every collection (DB cascade)
    * name 1-40 chars after trimming, unique per user (case-insensitive)
    * every query is scoped to the user - another user's ids give 404
"""

import logging
from datetime import datetime, timezone
from uuid import UUID

from postgrest.exceptions import APIError

from core.storage.supabase_client import supabase
from shared.exceptions import FitCheckException
from pipelines.tryon_looks_pipeline import _to_look

logger = logging.getLogger(__name__)

COLLECTIONS = "collections"
ENTRIES = "collection_entries"
MAX_NAME_LEN = 40
MAX_COLLECTIONS = 100
MAX_PAGE = 100
ENTRY_TYPES = {"look": "look_id", "outfit": "outfit_id"}


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
            "entry_type must be 'look' or 'outfit'.",
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
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        })
    return out


def _check_target_owned(uid: str, entry_type: str, entry_id: str) -> None:
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

    entries = supabase.table(ENTRIES).select("look_id, outfit_id, added_at")\
        .eq("collection_id", cid).eq("user_id", uid)\
        .order("added_at", desc=True)\
        .range(offset, offset + limit - 1)\
        .execute().data or []

    look_ids = [e["look_id"] for e in entries if e.get("look_id")]
    outfit_ids = [e["outfit_id"] for e in entries if e.get("outfit_id")]

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

    out = []
    for e in entries:
        if e.get("look_id") and e["look_id"] in looks:
            out.append({"entry_type": "look", "added_at": e["added_at"], "look": looks[e["look_id"]]})
        elif e.get("outfit_id") and e["outfit_id"] in outfits:
            out.append({"entry_type": "outfit", "added_at": e["added_at"], "outfit": outfits[e["outfit_id"]]})

    return {"entries": out, "limit": limit, "offset": offset}


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