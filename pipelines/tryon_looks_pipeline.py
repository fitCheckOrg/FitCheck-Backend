"""
Saved try-on looks.

A look = 1+ garments the user owns + one completed try-on render of theirs.
Separate from outfits on purpose: outfits keep their "at least 2 items"
rule; a look can be a single garment because the render IS the picture.

Table: public.tryon_looks
    unique (user_id, tryon_key)  -> saving the same render twice returns
                                    the existing look (idempotent), 200

Integrity checks on save:
    * tryon_key must be a COMPLETED render in tryon_renders owned by this user
    * the render's garment must be one of garment_ids
    * every garment id must be in this user's closet_items
"""

import logging
from uuid import UUID

from postgrest.exceptions import APIError

from core.storage.s3 import storage
from core.storage.supabase_client import supabase
from shared.exceptions import FitCheckException, ClothingItemNotFoundError

logger = logging.getLogger(__name__)

TABLE = "tryon_looks"
MAX_GARMENTS = 10
MAX_NAME_LEN = 80
MAX_PAGE = 100


def _bad_request(message: str, code: str) -> FitCheckException:
    return FitCheckException(message, code=code, status_code=400)


def _to_look(row: dict) -> dict:
    return {
        "look_id": row["look_id"],
        "garment_ids": row["garment_ids"],
        "tryon_key": row["tryon_key"],
        "image_url": storage.get_url(row["tryon_key"]),
        "name": row.get("name"),
        "created_at": row["created_at"],
    }


def _find_by_key(user_id: str, tryon_key: str):
    res = supabase.table(TABLE).select("*")\
        .eq("user_id", user_id).eq("tryon_key", tryon_key)\
        .limit(1).execute()
    return res.data[0] if res.data else None


def save_look(user_id: UUID, garment_ids: list[UUID], tryon_key: str,
              name: str | None = None) -> tuple[dict, bool]:
    """Returns (look, created). created=False when this render was already saved."""
    uid = str(user_id)

    # --- validate input
    ids = list(dict.fromkeys(str(g) for g in garment_ids))  # dedupe, keep order
    if not ids:
        raise _bad_request("A look needs at least 1 item.", "LOOK_NEEDS_ITEM")
    if len(ids) > MAX_GARMENTS:
        raise _bad_request(f"A look can have at most {MAX_GARMENTS} items.", "LOOK_TOO_MANY_ITEMS")

    key = (tryon_key or "").strip()
    if not key.startswith("tryons/"):
        raise _bad_request("tryon_key must be a try-on render key (tryons/...).", "INVALID_TRYON_KEY")

    clean_name = (name or "").strip() or None
    if clean_name and len(clean_name) > MAX_NAME_LEN:
        raise _bad_request(f"Name must be {MAX_NAME_LEN} characters or fewer.", "LOOK_NAME_TOO_LONG")

    # --- the render must exist, be finished, and belong to this user
    render = supabase.table("tryon_renders").select("garment_id")\
        .eq("user_id", uid).eq("s3_key", key).eq("status", "completed")\
        .limit(1).execute()
    if not render.data:
        raise FitCheckException(
            "Try-on render not found.", code="TRYON_RENDER_NOT_FOUND", status_code=404
        )
    if str(render.data[0]["garment_id"]) not in ids:
        raise _bad_request(
            "The try-on render is not of any of the given items.", "LOOK_GARMENT_MISMATCH"
        )

    # --- every garment must be in this user's closet
    owned = supabase.table("closet_items").select("item_id")\
        .eq("user_id", uid).in_("item_id", ids).execute()
    if len({str(r["item_id"]) for r in (owned.data or [])}) != len(ids):
        raise ClothingItemNotFoundError()

    # --- idempotent insert
    existing = _find_by_key(uid, key)
    if existing:
        return _to_look(existing), False

    row = {"user_id": uid, "tryon_key": key, "garment_ids": ids, "name": clean_name}
    try:
        res = supabase.table(TABLE).insert(row).execute()
        logger.info("saved look user=%s key=%s items=%d", uid, key, len(ids))
        return _to_look(res.data[0]), True
    except APIError as exc:
        if getattr(exc, "code", None) == "23505":  # saved by a parallel request
            existing = _find_by_key(uid, key)
            if existing:
                return _to_look(existing), False
        raise


def list_looks(user_id: UUID, limit: int = 50, offset: int = 0) -> dict:
    limit = max(1, min(int(limit), MAX_PAGE))
    offset = max(0, int(offset))
    res = supabase.table(TABLE).select("*")\
        .eq("user_id", str(user_id))\
        .order("created_at", desc=True)\
        .range(offset, offset + limit - 1)\
        .execute()
    return {"looks": [_to_look(r) for r in (res.data or [])], "limit": limit, "offset": offset}


def delete_look(user_id: UUID, look_id: UUID) -> dict:
    res = supabase.table(TABLE).delete()\
        .eq("look_id", str(look_id)).eq("user_id", str(user_id))\
        .execute()
    if not res.data:
        raise FitCheckException("Look not found.", code="LOOK_NOT_FOUND", status_code=404)
    return {"look_id": str(look_id), "deleted": True}


def rename_look(user_id: UUID, look_id: UUID, name: str | None) -> dict:
    """Set or clear (null / empty string) a look's name."""
    clean_name = (name or "").strip() or None
    if clean_name and len(clean_name) > MAX_NAME_LEN:
        raise _bad_request(f"Name must be {MAX_NAME_LEN} characters or fewer.", "LOOK_NAME_TOO_LONG")

    res = supabase.table(TABLE).update({"name": clean_name})\
        .eq("look_id", str(look_id)).eq("user_id", str(user_id))\
        .execute()
    if not res.data:
        raise FitCheckException("Look not found.", code="LOOK_NOT_FOUND", status_code=404)
    return _to_look(res.data[0])