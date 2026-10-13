"""
Try-on rendering pipeline — V2 (learned try-on model + render cache).

    avatar row + garment row
        -> cache lookup in tryon_renders
        -> completed?           return cached image (no model call)
        -> someone rendering?   return {"status": "processing"}
        -> otherwise            claim row, render (fal/FASHN), upload to S3,
                                mark completed, return image

Cache identity:
    (avatar_id, avatar_version, garment_id, asset_revision,
     provider, model, renderer_version)
A new avatar upload bumps avatar_version, an edited garment changes
asset_revision, a new model/settings bumps RENDERER_VERSION — each one
misses the cache and renders fresh. Old rows are simply never matched.

Rules the table design forces on us (agreed when the table was made):
  * a 'processing' row older than STALE_AFTER is treated as dead
    (server crashed mid-render) and may be re-claimed;
  * a retry after 'failed' UPDATES the existing row, never inserts.

The V1 mesh path (FittingEngine / Compositor / garment landmarks) is not
used here. It stays in the repo, frozen.
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


def _render_and_store(row_id: str, avatar: dict, garment: dict, category: str) -> dict:
    """Blocking (~15 s). Always called through run_in_threadpool."""
    t0 = time.time()
    try:
        png = get_renderer().render(
            avatar["original_photo_url"], garment["clean_image_url"], category
        )
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


# ------------------------------------------------------------------ entry point

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

        key = {
            "avatar_id": avatar["avatar_id"],
            "avatar_version": avatar["avatar_version"],
            "garment_id": gid,
            "asset_revision": f"md5:{garment['image_hash']}",
            "provider": PROVIDER,
            "model": MODEL,
            "renderer_version": RENDERER_VERSION,
        }

        row = _find_row(key)
        if row and row["status"] == "completed" and row.get("s3_key"):
            logger.info("try-on cache hit user=%s garment=%s", uid, gid)
            return {
                "status": "completed",
                "url": storage.get_url(row["s3_key"]),
                "key": row["s3_key"],
                "cached": True,
            }

        row_id = _claim(row, key, uid)
        if row_id is None:
            logger.info("try-on already processing user=%s garment=%s", uid, gid)
            return {"status": "processing"}

        logger.info("try-on cache miss, rendering user=%s garment=%s category=%s", uid, gid, category)
        return await run_in_threadpool(_render_and_store, row_id, avatar, garment, category)

    except FitCheckException:
        raise

    except Exception as exc:
        logger.exception("Try-on pipeline failed — user=%s garment=%s", uid, gid)
        raise FitCheckException(
            "Try-on rendering failed.",
            code="TRYON_RENDER_ERROR",
            status_code=500,
        ) from exc