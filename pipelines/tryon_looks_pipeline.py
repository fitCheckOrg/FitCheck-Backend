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

Try-On Studio B16 (2026-11-26): BOARD looks - saved without spending a render.
    POST /api/tryon/looks {"outfit_id", "name"?, "layout"?}      (no image_id)
      -> a look with kind "board", image null, tryon_key null, image_url null.
      * one board look per outfit: saving the same outfit again returns it (200)
        (a sent layout / name updates it); an outfit already saved WITH a picture
        returns that look.
      * later "See it on me" + POST {outfit_id, image_id} FILLS that board look's
        picture (same look_id, kind becomes "photo", 200).
    Every look carries "kind" ("photo" | "board") and "layout" (the app's board
    layout, <= 4 KB; copied from the outfit when the save doesn't send one).
    PATCH /api/tryon/looks/{look_id} {"name"?, "layout"?}: only the fields sent change.
    Deleting the avatar keeps board looks (no picture of the person in them).
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
    image = None
    if row.get("tryon_key"):
        signed = storage.signed(row["tryon_key"])
        image = {"image_id": str(row["render_id"]) if row.get("render_id") else None,
                 "key": signed["key"], "url": signed["url"], "expires_at": signed["expires_at"],
                 "outdated": row.get("outdated")}
    return {
        "look_id": row["look_id"],
        "kind": "photo" if image else "board",
        "outfit_id": str(row["outfit_id"]) if row.get("outfit_id") else None,
        "garment_ids": row["garment_ids"],
        "tryon_key": row.get("tryon_key"),
        "image_url": image["url"] if image else None,
        "image": image,
        "layout": row.get("layout"),
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


def _board_look(uid: str, outfit_id: str) -> dict | None:
    """B16: the board-only look of this outfit, if any."""
    rows = supabase.table(TABLE).select("*").eq("user_id", uid).eq("outfit_id", outfit_id)\
        .is_("tryon_key", "null").limit(1).execute().data
    return rows[0] if rows else None


def _store(uid: str, ids: list[str], render: dict, name: str | None,
           outfit_id: str | None, layout=None, layout_sent: bool = False) -> tuple[dict, bool]:
    from pipelines.tryon_outfits_pipeline import outfit_layout
    key = render["s3_key"]
    existing = _find_by_key(uid, key)
    if existing:
        if outfit_id and not existing.get("outfit_id"):
            res = supabase.table(TABLE).update({"outfit_id": outfit_id})\
                .eq("look_id", existing["look_id"]).eq("user_id", uid).execute()
            if res.data:
                existing = res.data[0]
        return _to_look(existing), False

    if outfit_id:
        board = _board_look(uid, outfit_id)
        if board:                                  # B16: fill the board look's picture
            fields = {"tryon_key": key, "render_id": render["id"], "garment_ids": ids}
            if name:
                fields["name"] = name
            if layout_sent:
                fields["layout"] = layout
            res = supabase.table(TABLE).update(fields)\
                .eq("look_id", board["look_id"]).eq("user_id", uid).execute()
            if res.data:
                logger.info("board look %s got its picture user=%s key=%s", board["look_id"], uid, key)
                return _to_look(res.data[0]), False

    if not layout_sent and outfit_id:
        layout = outfit_layout(uid, outfit_id)
    row = {"user_id": uid, "tryon_key": key, "garment_ids": ids, "name": name,
           "render_id": render["id"], "outfit_id": outfit_id, "layout": layout}
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


def _save_board(uid: str, oid: str, ids: list[str], name: str | None,
                layout, layout_sent: bool) -> tuple[dict, bool]:
    """B16: a look without a render. One per outfit; an outfit already saved with a
    picture returns that look."""
    from pipelines.tryon_outfits_pipeline import outfit_layout
    rows = supabase.table(TABLE).select("*").eq("user_id", uid).eq("outfit_id", oid)\
        .order("created_at", desc=True).execute().data or []
    if rows:
        existing = next((r for r in rows if not r.get("tryon_key")), rows[0])
        fields = {}
        if name:
            fields["name"] = name
        if layout_sent:
            fields["layout"] = layout
        if fields:
            res = supabase.table(TABLE).update(fields)\
                .eq("look_id", existing["look_id"]).eq("user_id", uid).execute()
            if res.data:
                existing = res.data[0]
        return _to_look(existing), False

    row = {"user_id": uid, "tryon_key": None, "render_id": None, "garment_ids": ids,
           "name": name, "outfit_id": oid,
           "layout": layout if layout_sent else outfit_layout(uid, oid)}
    try:
        res = supabase.table(TABLE).insert(row).execute()
        logger.info("saved board look user=%s outfit=%s items=%d", uid, oid, len(ids))
        return _to_look(res.data[0]), True
    except APIError as exc:
        if getattr(exc, "code", None) == "23505":  # saved by a parallel request
            board = _board_look(uid, oid)
            if board:
                return _to_look(board), False
        raise


def save_look(user_id: UUID, garment_ids: list[UUID] | None = None, tryon_key: str | None = None,
              name: str | None = None, outfit_id: UUID | None = None,
              image_id: UUID | None = None, layout=None,
              layout_sent: bool = False) -> tuple[dict, bool]:
    """Returns (look, created). created=False when this render / board was already saved."""
    from pipelines.tryon_outfits_pipeline import clean_layout
    uid = str(user_id)
    clean_name = _clean_name(name)
    layout = clean_layout(layout) if layout_sent else None

    # --- B7b: {outfit_id, image_id}  ·  B16: {outfit_id} alone = a board look
    if outfit_id is not None or image_id is not None:
        if outfit_id is None:
            raise _bad_request("Send outfit_id with image_id.", "LOOK_BODY_INVALID")
        oid, ids = _outfit_garments(uid, outfit_id)
        if not ids:
            raise _bad_request("A look needs at least 1 item.", "LOOK_NEEDS_ITEM")
        if len(ids) > MAX_GARMENTS:
            raise _bad_request(f"A look can have at most {MAX_GARMENTS} items.", "LOOK_TOO_MANY_ITEMS")
        if image_id is None:
            return _save_board(uid, oid, ids, clean_name, layout, layout_sent)
        render = _render_by_id(uid, image_id)
        if str(render["garment_id"]) not in ids:
            raise _bad_request("The try-on render is not of this outfit.", "LOOK_GARMENT_MISMATCH")
        return _store(uid, ids, render, clean_name, oid, layout, layout_sent)

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
    return update_look(user_id, look_id, {"name": name})


def update_look(user_id: UUID, look_id: UUID, fields: dict) -> dict:
    """B16: change only the fields sent - name (null / empty clears) and / or layout."""
    from pipelines.tryon_outfits_pipeline import clean_layout
    changes = {}
    if "name" in fields:
        changes["name"] = _clean_name(fields["name"])
    if "layout" in fields:
        changes["layout"] = clean_layout(fields["layout"])
    if not changes:
        raise _bad_request("Send name and / or layout.", "LOOK_BODY_INVALID")
    res = supabase.table(TABLE).update(changes)\
        .eq("look_id", str(look_id)).eq("user_id", str(user_id))\
        .execute()
    if not res.data:
        raise FitCheckException("Look not found.", code="LOOK_NOT_FOUND", status_code=404)
    return _to_look(res.data[0])