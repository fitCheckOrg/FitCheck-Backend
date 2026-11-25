"""
Saved try-on looks.

A look = 1+ garments the user owns + one completed try-on render of theirs.
Separate from outfits on purpose: a look can be a single garment because the
render IS the picture.

Table: public.tryon_looks
    unique (user_id, tryon_key)  -> saving the same render twice returns
                                    the existing look (idempotent), 200

Try-On v2 T2-1 B5 (2026-11-18): every look carries the image object
    image: {image_id, key, url, expires_at, outdated}
(url is a signed link for 1 hour; re-sign with POST /api/media/sign). image_url
stays (same signed link) and tryon_key stays until the app has switched.
image_id is the render's id (tryon_looks.render_id, set on save).
B6 (2026-11-18): outdated = "garment_deleted" | "avatar_replaced" | null. A look
always keeps its image file, even when outdated.

B7b (2026-11-25): save accepts {outfit_id, image_id, name?}:
    * the outfit must be the user's; its pieces that still exist become garment_ids
    * image_id must be a COMPLETED render (tryon_renders.id) owned by this user,
      of one of those pieces
    * the look stores outfit_id (tryon_looks.outfit_id) and returns it
The old body {garment_ids, tryon_key, name?} still works, with the same checks as before:
    * tryon_key must be a COMPLETED render in tryon_renders owned by this user
    * the render's garment must be one of garment_ids
    * every garment id must be in this user's closet_items
Saving a render that's already saved returns that look (200); if it had no
outfit_id yet and one was sent, it's filled in.
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


def _not_found(message: str, code: str) -> FitCheckException:
    return FitCheckException(message, code=code, status_code=404)


def _to_look(row: dict) -> dict:
    signed = storage.signed(row["tryon_key"])
    return {
        "look_id": row["look_id"],
        "outfit_id": str(row["outfit_id"]) if row.get("outfit_id") else None,
        "garment_ids": row["garment_ids"],
        "tryon_key": row["tryon_key"],
        "image_url": signed["url"],
        "image": {"image_id": str(row["render_id"]) if row.get("render_id") else None,
                  "key": signed["key"], "url": signed["url"], "expires_at": signed["expires_at"],
                  "outdated": row.get("outdated")},
        "name": row.get("name"),
        "created_at": row["created_at"],
    }


def _find_by_key(user_id: str, tryon_key: str):
    res = supabase.table(TABLE).select("*")\
        .eq("user_id", user_id).eq("tryon_key", tryon_key)\
        .limit(1).execute()
    return res.data[0] if res.data else None


def _clean_name(name: str | None) -> str | None:
    clean = (name or "").strip() or None
    if clean and len(clean) > MAX_NAME_LEN:
        raise _bad_request(f"Name must be {MAX_NAME_LEN} characters or fewer.", "LOOK_NAME_TOO_LONG")
    return clean


def _as_uuid(value, code: str, message: str) -> str:
    try:
        return str(UUID(str(value)))
    except Exception:
        raise _not_found(message, code)


def _owned_ids(uid: str, ids: list[str]) -> set[str]:
    if not ids:
        return set()
    owned = supabase.table("closet_items").select("item_id")\
        .eq("user_id", uid).in_("item_id", ids).execute()
    return {str(r["item_id"]) for r in (owned.data or [])}


def _outfit_garments(uid: str, outfit_id) -> tuple[str, list[str]]:
    """(outfit_id, ids of its pieces that still exist), in slot order."""
    from pipelines.tryon_slots import SLOT_ORDER
    oid = _as_uuid(outfit_id, "OUTFIT_NOT_FOUND", "Outfit not found.")
    head = supabase.table("tryon_outfits").select("outfit_id")\
        .eq("outfit_id", oid).eq("user_id", uid).limit(1).execute().data
    if not head:
        raise _not_found("Outfit not found.", "OUTFIT_NOT_FOUND")
    rows = supabase.table("tryon_outfit_items").select("slot, position, garment_id")\
        .eq("outfit_id", oid).execute().data or []
    rows.sort(key=lambda r: (SLOT_ORDER.index(r["slot"]) if r["slot"] in SLOT_ORDER else 99, r["position"]))
    ids = list(dict.fromkeys(str(r["garment_id"]) for r in rows))
    owned = _owned_ids(uid, ids)
    return oid, [g for g in ids if g in owned]


def _render_by_id(uid: str, image_id) -> dict:
    rid = _as_uuid(image_id, "TRYON_RENDER_NOT_FOUND", "Try-on render not found.")
    res = supabase.table("tryon_renders").select("id, garment_id, s3_key")\
        .eq("user_id", uid).eq("id", rid).eq("status", "completed")\
        .limit(1).execute()
    if not res.data or not res.data[0].get("s3_key"):
        raise _not_found("Try-on render not found.", "TRYON_RENDER_NOT_FOUND")
    return res.data[0]


def _render_by_key(uid: str, key: str) -> dict:
    res = supabase.table("tryon_renders").select("id, garment_id, s3_key")\
        .eq("user_id", uid).eq("s3_key", key).eq("status", "completed")\
        .limit(1).execute()
    if not res.data:
        raise _not_found("Try-on render not found.", "TRYON_RENDER_NOT_FOUND")
    return res.data[0]


def _store(uid: str, ids: list[str], render: dict, name: str | None,
           outfit_id: str | None) -> tuple[dict, bool]:
    key = render["s3_key"]
    existing = _find_by_key(uid, key)
    if existing:
        if outfit_id and not existing.get("outfit_id"):
            res = supabase.table(TABLE).update({"outfit_id": outfit_id})\
                .eq("look_id", existing["look_id"]).eq("user_id", uid).execute()
            if res.data:
                existing = res.data[0]
        return _to_look(existing), False

    row = {"user_id": uid, "tryon_key": key, "garment_ids": ids, "name": name,
           "render_id": render["id"], "outfit_id": outfit_id}
    try:
        res = supabase.table(TABLE).insert(row).execute()
        logger.info("saved look user=%s key=%s items=%d outfit=%s", uid, key, len(ids), outfit_id)
        return _to_look(res.data[0]), True
    except APIError as exc:
        if getattr(exc, "code", None) == "23505":  # saved by a parallel request
            existing = _find_by_key(uid, key)
            if existing:
                return _to_look(existing), False
        raise


def save_look(user_id: UUID, garment_ids: list[UUID] | None = None, tryon_key: str | None = None,
              name: str | None = None, outfit_id: UUID | None = None,
              image_id: UUID | None = None) -> tuple[dict, bool]:
    """Returns (look, created). created=False when this render was already saved."""
    uid = str(user_id)
    clean_name = _clean_name(name)

    # --- B7b: {outfit_id, image_id}
    if outfit_id is not None or image_id is not None:
        if outfit_id is None or image_id is None:
            raise _bad_request("Send both outfit_id and image_id.", "LOOK_BODY_INVALID")
        oid, ids = _outfit_garments(uid, outfit_id)
        if not ids:
            raise _bad_request("A look needs at least 1 item.", "LOOK_NEEDS_ITEM")
        if len(ids) > MAX_GARMENTS:
            raise _bad_request(f"A look can have at most {MAX_GARMENTS} items.", "LOOK_TOO_MANY_ITEMS")
        render = _render_by_id(uid, image_id)
        if str(render["garment_id"]) not in ids:
            raise _bad_request("The try-on render is not of this outfit.", "LOOK_GARMENT_MISMATCH")
        return _store(uid, ids, render, clean_name, oid)

    # --- old body: {garment_ids, tryon_key}
    if not garment_ids or not tryon_key:
        raise _bad_request("Send {outfit_id, image_id} or {garment_ids, tryon_key}.", "LOOK_BODY_INVALID")
    ids = list(dict.fromkeys(str(g) for g in garment_ids))  # dedupe, keep order
    if len(ids) > MAX_GARMENTS:
        raise _bad_request(f"A look can have at most {MAX_GARMENTS} items.", "LOOK_TOO_MANY_ITEMS")
    key = tryon_key.strip()
    if not key.startswith("tryons/"):
        raise _bad_request("tryon_key must be a try-on render key (tryons/...).", "INVALID_TRYON_KEY")
    render = _render_by_key(uid, key)
    if str(render["garment_id"]) not in ids:
        raise _bad_request("The try-on render is not of any of the given items.", "LOOK_GARMENT_MISMATCH")
    if len(_owned_ids(uid, ids)) != len(ids):
        raise ClothingItemNotFoundError()
    return _store(uid, ids, render, clean_name, None)


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
    clean_name = _clean_name(name)
    res = supabase.table(TABLE).update({"name": clean_name})\
        .eq("look_id", str(look_id)).eq("user_id", str(user_id))\
        .execute()
    if not res.data:
        raise FitCheckException("Look not found.", code="LOOK_NOT_FOUND", status_code=404)
    return _to_look(res.data[0])
