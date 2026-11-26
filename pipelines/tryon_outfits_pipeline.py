"""
Try-On v2 T2-1 B1 (2026-11-17): outfits - the chosen garments, by slot, with
optional fit preferences. An outfit is a VALUE: the same content always gives
the same outfit_id (tryon_outfits.unique(user_id, content_hash)), so the
dressing room can create outfits freely while experimenting.

POST /api/tryon/outfits  {items: [{garment_id, slot?, fit?}]}
GET  /api/tryon/outfits/{outfit_id}
  -> {outfit_id, items: [{slot, position, garment_id, fit, renders, missing}],
      rendered_slots, tray_slots, needs_generations, cached_image, created_at}

needs_generations = provider renders a "See it on me" would need right now, for the
user's CURRENT avatar (0 = already rendered / nothing to render; null = no avatar).
cached_image = the finished render of exactly the rendered pieces, or null.
Fit preferences never change the render (styling preference only).

B7b (2026-11-25): play-mode style values (Keji's fit principles). fit = any of
    silhouette: slim | regular | relaxed | oversized     (absent = as the garment is)
    length:     cropped | regular | long                 (absent = as the garment is)
    styling:    tucked | untucked | front_tuck | layered (absent = default, untucked for tops)
    rise:       low | mid | high                         (absent = the garment's own rise)
"front-tuck" is accepted and stored as "front_tuck". Existing outfits are unchanged.

Try-On Studio B16 (2026-11-26): the board layout - an opaque JSON object the app owns
(nudges, board surface, ...), at most 4 KB, stored with the outfit:
    PUT /api/tryon/outfits/{outfit_id}/layout  {"layout": {...} | null}   -> outfit
    every outfit returns "layout" (null when none was saved).
The layout never changes the outfit id, the render or its cost.

Tables (T2-1 B1 SQL): tryon_outfits, tryon_outfit_items; function create_tryon_outfit.
"""

import hashlib
import json
import logging
from uuid import UUID

from core.storage.s3 import storage
from core.storage.supabase_client import supabase
from shared.exceptions import FitCheckException, ClothingItemNotFoundError
from pipelines.tryon_slots import SLOTS, SLOT_ORDER, allowed_slots, slot_for_category

logger = logging.getLogger(__name__)

MAX_ITEMS = 10
MAX_LAYOUT_BYTES = 4096
FIT_VALUES = {
    "silhouette": ("slim", "regular", "relaxed", "oversized"),
    "length": ("cropped", "regular", "long"),
    "styling": ("tucked", "untucked", "front_tuck", "layered"),
    "rise": ("low", "mid", "high"),
}


def _bad(code: str, message: str, status: int = 400) -> FitCheckException:
    return FitCheckException(message, code=code, status_code=status)


# ---------------------------------------------------------------- validation

def _clean_fit(raw) -> dict | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise _bad("FIT_INVALID", "fit must be an object.")
    out = {}
    for key, value in raw.items():
        if key not in FIT_VALUES:
            raise _bad("FIT_INVALID", f"Unknown fit setting: {key}. Allowed: {', '.join(FIT_VALUES)}.")
        if value is None:
            continue
        v = str(value).strip().lower().replace("-", "_")
        if v not in FIT_VALUES[key]:
            raise _bad("FIT_INVALID", f"{key} must be one of: {', '.join(FIT_VALUES[key])}.")
        out[key] = v
    return {k: out[k] for k in sorted(out)} or None


def clean_layout(raw) -> dict | None:
    """B16: the app's board layout - any JSON object up to 4 KB, or null."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise _bad("LAYOUT_INVALID", "layout must be a JSON object (or null).")
    size = len(json.dumps(raw, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    if size > MAX_LAYOUT_BYTES:
        raise _bad("LAYOUT_TOO_BIG", f"layout must be {MAX_LAYOUT_BYTES} bytes or less (it is {size}).")
    return raw


def outfit_layout(uid: str, outfit_id: str) -> dict | None:
    rows = supabase.table("tryon_outfits").select("layout")\
        .eq("outfit_id", outfit_id).eq("user_id", uid).limit(1).execute().data
    return rows[0].get("layout") if rows else None


def _load_garments(uid: str, ids: list[str]) -> dict[str, dict]:
    if not ids:
        return {}
    rows = supabase.table("closet_items")\
        .select("item_id, category, image_hash, clean_image_url")\
        .eq("user_id", uid).in_("item_id", ids).execute().data or []
    return {str(r["item_id"]): r for r in rows}


# ---------------------------------------------------------------- render plan

def _plan(items: list[dict]) -> tuple[list[dict], list[str], list[str]]:
    """(pieces that render in chain order, rendered_slots, tray_slots). Sets item['renders']."""
    present = [i for i in items if not i["missing"]]
    first = {}
    for i in present:
        first.setdefault(i["slot"], i)
    if "one_piece" in first:
        rendered = [first["one_piece"]]
    else:
        upper = first.get("top") or first.get("outerwear")
        rendered = [p for p in (upper, first.get("bottom")) if p]
    ids = {id(p) for p in rendered}
    for i in items:
        i["renders"] = id(i) in ids
    tray = []
    for s in SLOT_ORDER:
        if any(i["slot"] == s and not i["renders"] for i in present) and s not in tray:
            tray.append(s)
    return rendered, [p["slot"] for p in rendered], tray


def _image(row: dict, avatar: dict, garment_ids: list[str]) -> dict:
    """Image object. B5: url is a signed link until expires_at (re-sign: POST /api/media/sign).
    B6: outdated = "garment_deleted" | "avatar_replaced" | null; url null when the file is gone."""
    signed = storage.signed(row["s3_key"])
    gone = bool(row.get("file_deleted_at"))
    return {"image_id": str(row["id"]), "key": signed["key"],
            "url": None if gone else signed["url"], "expires_at": None if gone else signed["expires_at"],
            "avatar_version": avatar["avatar_version"], "garment_ids": garment_ids,
            "outdated": row.get("outdated")}


def _cost(uid: str, rendered: list[dict]) -> tuple[int | None, dict | None]:
    """(needs_generations, cached_image) for the user's current avatar."""
    from pipelines.tryon_pipeline import _single_key, _chain_key, _find_row
    if not rendered:
        return 0, None
    av = supabase.table("avatars").select("avatar_id, avatar_version")\
        .eq("user_id", uid).limit(1).execute().data
    if not av:
        return None, None
    avatar = av[0]
    g = [p["garment"] for p in rendered]
    garment_ids = [str(x["item_id"]) for x in g]
    keys = [_single_key(avatar, g[0])]
    if len(g) == 2:
        keys.append(_chain_key(avatar, g[0], g[1]))
    rows = [_find_row(k) for k in keys]
    done = [bool(r and r.get("status") == "completed" and r.get("s3_key")) for r in rows]
    if done[-1]:
        return 0, _image(rows[-1], avatar, garment_ids)
    if len(done) == 2 and done[0]:
        return 1, None
    return len(keys), None


# ---------------------------------------------------------------- read

def get_outfit(user_id, outfit_id) -> dict:
    uid = str(user_id)
    try:
        oid = str(UUID(str(outfit_id)))
    except Exception:
        raise _bad("OUTFIT_NOT_FOUND", "Outfit not found.", 404)
    head = supabase.table("tryon_outfits").select("outfit_id, created_at, layout")\
        .eq("outfit_id", oid).eq("user_id", uid).limit(1).execute().data
    if not head:
        raise _bad("OUTFIT_NOT_FOUND", "Outfit not found.", 404)
    rows = supabase.table("tryon_outfit_items").select("slot, position, garment_id, fit")\
        .eq("outfit_id", oid).execute().data or []
    garments = _load_garments(uid, [str(r["garment_id"]) for r in rows])
    items = []
    for r in sorted(rows, key=lambda r: (SLOT_ORDER.index(r["slot"]) if r["slot"] in SLOT_ORDER else 99,
                                         r["position"])):
        gid = str(r["garment_id"])
        items.append({"slot": r["slot"], "position": r["position"], "garment_id": gid,
                      "fit": r.get("fit"), "garment": garments.get(gid), "missing": gid not in garments})
    rendered, rendered_slots, tray_slots = _plan(items)
    needs, cached = _cost(uid, rendered)
    return {
        "outfit_id": oid,
        "items": [{k: i[k] for k in ("slot", "position", "garment_id", "fit", "renders", "missing")}
                  for i in items],
        "rendered_slots": rendered_slots,
        "tray_slots": tray_slots,
        "needs_generations": needs,
        "cached_image": cached,
        "layout": head[0].get("layout"),
        "created_at": head[0]["created_at"],
    }


def set_layout(user_id, outfit_id, layout) -> dict:
    """B16: save (or clear with null) the board layout of one of the user's outfits."""
    uid = str(user_id)
    try:
        oid = str(UUID(str(outfit_id)))
    except Exception:
        raise _bad("OUTFIT_NOT_FOUND", "Outfit not found.", 404)
    clean = clean_layout(layout)
    res = supabase.table("tryon_outfits").update({"layout": clean})\
        .eq("outfit_id", oid).eq("user_id", uid).execute()
    if not res.data:
        raise _bad("OUTFIT_NOT_FOUND", "Outfit not found.", 404)
    return get_outfit(uid, oid)


# ---------------------------------------------------------------- create

def create_outfit(user_id, items: list[dict]) -> dict:
    """Same content -> same outfit_id. Returns the outfit."""
    uid = str(user_id)
    if not items:
        raise _bad("OUTFIT_EMPTY", "Pick at least one piece.")

    seen, wanted = set(), []
    for raw in items:
        gid = str(raw["garment_id"])
        if gid in seen:
            continue                                   # same piece twice -> once
        seen.add(gid)
        wanted.append({"garment_id": gid, "slot": raw.get("slot"), "fit": _clean_fit(raw.get("fit"))})
    if len(wanted) > MAX_ITEMS:
        raise _bad("SLOT_CONFLICT", f"An outfit can have at most {MAX_ITEMS} pieces.")

    garments = _load_garments(uid, [w["garment_id"] for w in wanted])
    if len(garments) != len(wanted):
        raise ClothingItemNotFoundError()

    by_slot: dict[str, list[dict]] = {}
    for w in wanted:
        category = garments[w["garment_id"]]["category"]
        slot = str(w["slot"]).strip().lower() if w["slot"] else slot_for_category(category)
        if not slot or slot not in allowed_slots(category):
            raise _bad("SLOT_INVALID", f"A {category} piece can't go in the {slot or 'unknown'} slot.")
        by_slot.setdefault(slot, []).append(w)

    for slot, pieces in by_slot.items():
        if len(pieces) > SLOTS[slot]["max"]:
            raise _bad("SLOT_CONFLICT", f"Only {SLOTS[slot]['max']} piece(s) fit in {slot}.")
        for other in SLOTS[slot]["excludes"]:
            if other in by_slot:
                raise _bad("SLOT_CONFLICT", f"{slot} can't be worn with {other}.")

    canonical = []
    for slot in SLOT_ORDER:
        for pos, w in enumerate(sorted(by_slot.get(slot, []), key=lambda w: w["garment_id"])):
            canonical.append({"slot": slot, "position": pos, "garment_id": w["garment_id"], "fit": w["fit"]})
    content_hash = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()

    res = supabase.rpc("create_tryon_outfit", {"p_user": uid, "p_hash": content_hash,
                                               "p_items": canonical}).execute()
    oid = res.data[0] if isinstance(res.data, list) else res.data
    logger.info("outfit user=%s outfit=%s pieces=%s", uid, oid, len(canonical))
    return get_outfit(uid, oid)