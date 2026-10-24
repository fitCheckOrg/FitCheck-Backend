"""
Try-on rendering pipeline - V2 (learned try-on model + render cache).

Single garment:   run_tryon_pipeline(garment_id, user_id)
Full outfit:      run_outfit_tryon_pipeline(garment_ids, user_id)
                  = chained renders: top onto the user's photo, then the
                  bottom onto THAT result. Each step is cached on its own,
                  so step 1 is shared with the single-garment try-on.

    cache lookup in tryon_renders
        -> completed?           return cached image (no model call)
        -> someone rendering?   return {"status": "processing"}
        -> otherwise            claim row, render (fal/FASHN), upload to S3,
                                mark completed, return image

Cache identity:
    (avatar_id, avatar_version, garment_id, asset_revision,
     provider, model, renderer_version)
A new avatar upload bumps avatar_version, an edited garment changes
asset_revision, a new model/settings bumps RENDERER_VERSION - each one
misses the cache and renders fresh. Old rows are simply never matched.
Chain steps use renderer_version "<v>-chain" and an asset_revision that
includes the first garment, so they never collide with single renders.

Rules the table design forces on us:
  * a 'processing' row older than STALE_AFTER is treated as dead
    (server crashed mid-render) and may be re-claimed;
  * a retry after 'failed' UPDATES the existing row, never inserts.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from uuid import UUID

from postgrest.exceptions import APIError
from starlette.concurrency import run_in_threadpool

from core.storage.s3 import storage, S3Folders
from core.storage.supabase_client import supabase
from shared.exceptions import (
    FitCheckException,
    AvatarNotFoundError,
    ClothingItemNotFoundError,
)
from tryon.vto_renderer import get_renderer, TryOnRenderError

logger = logging.getLogger(__name__)

TABLE = "tryon_renders"

# Must describe what get_renderer() actually returns.
# Change model/settings in vto_renderer.py  ->  bump RENDERER_VERSION here.
PROVIDER = "fal"
MODEL = "fashn-v1.6"
RENDERER_VERSION = "v1"
CHAIN_RENDERER_VERSION = f"{RENDERER_VERSION}-chain"

STALE_AFTER = timedelta(minutes=5)


# ------------------------------------------------------------------ helpers

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _unsupported(raw_category) -> FitCheckException:
    return FitCheckException(
        f"Try-on isn't available for '{raw_category}' items yet.",
        code="GARMENT_UNSUPPORTED",
        status_code=409,
    )


# Order matters: one-pieces before bottoms before tops.
_CATEGORY_RULES = [
    ("one-pieces", ("dress", "jumpsuit", "romper", "one-piece", "onepiece", "overall")),
    ("bottoms", ("bottom", "pant", "trouser", "jean", "short", "skirt", "legging", "jogger")),
    ("tops", ("top", "shirt", "tee", "blouse", "sweater", "hoodie", "jacket",
              "coat", "outerwear", "cardigan", "polo", "vest", "jersey")),
]


def _fashn_category(raw_category) -> str:
    c = (raw_category or "").strip().lower()
    for fashn_cat, keywords in _CATEGORY_RULES:
        if any(k in c for k in keywords):
            return fashn_cat
    raise _unsupported(raw_category)  # shoes, bags, accessories, unknown


def _load_avatar(user_id: str) -> dict:
    res = supabase.table("avatars")\
        .select("avatar_id, avatar_version, original_photo_url")\
        .eq("user_id", user_id)\
        .maybe_single()\
        .execute()
    if not res or not res.data:
        raise AvatarNotFoundError()
    return res.data


def _load_garment(garment_id: str, user_id: str) -> dict:
    res = supabase.table("closet_items")\
        .select("item_id, category, clean_image_url, image_hash")\
        .eq("item_id", garment_id)\
        .eq("user_id", user_id)\
        .maybe_single()\
        .execute()
    if not res or not res.data:
        raise ClothingItemNotFoundError()
    return res.data


def _single_key(avatar: dict, garment: dict) -> dict:
    return {
        "avatar_id": avatar["avatar_id"],
        "avatar_version": avatar["avatar_version"],
        "garment_id": str(garment["item_id"]),
        "asset_revision": f"md5:{garment['image_hash']}",
        "provider": PROVIDER,
        "model": MODEL,
        "renderer_version": RENDERER_VERSION,
    }


def _chain_key(avatar: dict, first: dict, second: dict) -> dict:
    return {
        "avatar_id": avatar["avatar_id"],
        "avatar_version": avatar["avatar_version"],
        "garment_id": str(second["item_id"]),
        "asset_revision": (
            f"chain:{first['item_id']}:md5:{first['image_hash']}"
            f">md5:{second['image_hash']}"
        ),
        "provider": PROVIDER,
        "model": MODEL,
        "renderer_version": CHAIN_RENDERER_VERSION,
    }


def _find_row(key: dict):
    q = supabase.table(TABLE).select("id, status, s3_key, updated_at")
    for col, val in key.items():
        q = q.eq(col, val)
    res = q.limit(1).execute()
    return res.data[0] if res.data else None


def _claim(row, key: dict, user_id: str):
    """
    Try to become the request that renders this key.
    Returns the row id if we own the render, None if someone else does.
    """
    now = _now_iso()
    claim = {"status": "processing", "error_message": None, "updated_at": now}

    if row is None:
        try:
            res = supabase.table(TABLE).insert({**key, "user_id": user_id, **claim}).execute()
            return res.data[0]["id"]
        except APIError as exc:
            if getattr(exc, "code", None) == "23505":  # unique violation: lost the race
                return None
            raise

    if row["status"] == "failed":
        res = supabase.table(TABLE).update(claim)\
            .eq("id", row["id"]).eq("status", "failed").execute()
    else:  # 'processing' / 'pending': only re-claim if stale
        cutoff = (datetime.now(timezone.utc) - STALE_AFTER).isoformat()
        res = supabase.table(TABLE).update(claim)\
            .eq("id", row["id"]).in_("status", ["processing", "pending"])\
            .lt("updated_at", cutoff).execute()

    return row["id"] if res.data else None


def _mark_failed(row_id: str, message: str):
    # Log the reason too: a successful retry clears error_message in the row.
    logger.warning("try-on render failed row=%s: %s", row_id, message)
    try:
        supabase.table(TABLE).update({
            "status": "failed",
            "error_message": message[:500],
            "updated_at": _now_iso(),
        }).eq("id", row_id).execute()
    except Exception:
        logger.exception("could not mark try-on render %s as failed", row_id)


def _render_and_store(row_id: str, person_url: str, garment_url: str, category: str) -> dict:
    """Blocking (~15 s). Always called through run_in_threadpool."""
    t0 = time.time()
    try:
        png = get_renderer().render(person_url, garment_url, category)
        uploaded = storage.upload(
            png, folder=S3Folders.TRYON, extension="png", content_type="image/png"
        )
    except Exception as exc:
        _mark_failed(row_id, str(exc))
        if isinstance(exc, TryOnRenderError):
            raise FitCheckException(
                "Try-on rendering failed.", code="TRYON_RENDER_ERROR", status_code=502
            ) from exc
        raise

    now = _now_iso()
    try:
        supabase.table(TABLE).update({
            "status": "completed",
            "s3_key": uploaded["key"],
            "error_message": None,
            "updated_at": now,
            "completed_at": now,
        }).eq("id", row_id).execute()
    except Exception:
        # Image exists but the row didn't update. It goes stale after
        # STALE_AFTER and gets re-rendered; the user still gets this image now.
        logger.exception("render %s uploaded but cache row not marked completed", row_id)

    logger.info("try-on rendered row=%s key=%s %.1fs", row_id, uploaded["key"], time.time() - t0)
    return {"status": "completed", "url": uploaded["url"], "key": uploaded["key"], "cached": False}


async def _ensure_render(user_id: str, key: dict, person_url: str,
                         garment_url: str, category: str) -> dict:
    """Cached image for this key, or render it now, or report 'processing'."""
    row = _find_row(key)
    if row and row["status"] == "completed" and row.get("s3_key"):
        return {
            "status": "completed",
            "url": storage.get_url(row["s3_key"]),
            "key": row["s3_key"],
            "cached": True,
        }

    row_id = _claim(row, key, user_id)
    if row_id is None:
        return {"status": "processing"}

    return await run_in_threadpool(_render_and_store, row_id, person_url, garment_url, category)


def _wrap_unexpected(exc: Exception, user_id: str, what: str) -> FitCheckException:
    logger.exception("Try-on pipeline failed - user=%s %s", user_id, what)
    return FitCheckException(
        "Try-on rendering failed.",
        code="TRYON_RENDER_ERROR",
        status_code=500,
    )


# ------------------------------------------------------------------ single garment

async def run_tryon_pipeline(garment_id: UUID, user_id: UUID) -> dict:
    """
    Returns one of:
        {"status": "completed", "url", "key", "cached": bool}
        {"status": "processing"}      -> route answers 202, client retries
    """
    uid, gid = str(user_id), str(garment_id)
    try:
        avatar = _load_avatar(uid)
        garment = _load_garment(gid, uid)
        category = _fashn_category(garment["category"])

        result = await _ensure_render(
            uid, _single_key(avatar, garment),
            avatar["original_photo_url"], garment["clean_image_url"], category,
        )
        logger.info("try-on %s user=%s garment=%s category=%s",
                    "cache hit" if result.get("cached") else result["status"], uid, gid, category)
        return result

    except FitCheckException:
        raise
    except Exception as exc:
        raise _wrap_unexpected(exc, uid, f"garment={gid}") from exc


# ------------------------------------------------------------------ full outfit (chained)

def _outfit_unsupported(message: str) -> FitCheckException:
    return FitCheckException(message, code="TRYON_OUTFIT_UNSUPPORTED", status_code=400)


async def run_outfit_tryon_pipeline(garment_ids: list[UUID], user_id: UUID) -> dict:
    """
    One top + one bottom -> chained render. A single garment (incl. a dress)
    behaves exactly like run_tryon_pipeline.

    Returns one of:
        {"status": "completed", "url", "key", "cached", "steps": [{garment_id, key}, ...]}
        {"status": "processing", "step": 1 | 2}
    """
    uid = str(user_id)
    ids = list(dict.fromkeys(str(g) for g in garment_ids))
    if not ids:
        raise _outfit_unsupported("Pick at least one item to try on.")
    if len(ids) > 2:
        raise _outfit_unsupported("Outfit try-on supports one top and one bottom for now.")

    try:
        avatar = _load_avatar(uid)
        garments = [_load_garment(g, uid) for g in ids]
        cats = [_fashn_category(g["category"]) for g in garments]  # 409 for shoes etc.

        if len(garments) == 2:
            if sorted(cats) != ["bottoms", "tops"]:
                raise _outfit_unsupported(
                    "Outfit try-on needs one top and one bottom. Dresses are tried on alone."
                )
            if cats[0] == "bottoms":            # always render the top first
                garments.reverse()
                cats.reverse()

        first = garments[0]
        step1 = await _ensure_render(
            uid, _single_key(avatar, first),
            avatar["original_photo_url"], first["clean_image_url"], cats[0],
        )
        if step1["status"] != "completed":
            return {"status": "processing", "step": 1}

        steps = [{"garment_id": str(first["item_id"]), "key": step1["key"]}]
        if len(garments) == 1:
            return {**step1, "steps": steps}

        second = garments[1]
        step2 = await _ensure_render(
            uid, _chain_key(avatar, first, second),
            step1["url"], second["clean_image_url"], cats[1],
        )
        if step2["status"] != "completed":
            return {"status": "processing", "step": 2}

        steps.append({"garment_id": str(second["item_id"]), "key": step2["key"]})
        logger.info("outfit try-on user=%s steps=%s cached=%s/%s",
                    uid, [s["garment_id"] for s in steps], step1["cached"], step2["cached"])
        return {
            "status": "completed",
            "url": step2["url"],
            "key": step2["key"],
            "cached": bool(step1["cached"] and step2["cached"]),
            "steps": steps,
        }

    except FitCheckException:
        raise
    except Exception as exc:
        raise _wrap_unexpected(exc, uid, f"outfit={ids}") from exc