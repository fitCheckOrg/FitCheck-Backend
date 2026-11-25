"""
Try-on render cache helpers - V2.

Since T2-1 B7b (2026-11-25) all rendering happens in the background worker
(pipelines/render_worker.py, render jobs). The old blocking routes
POST /api/tryon and /api/tryon/outfit and their pipeline functions
(run_tryon_pipeline, run_outfit_tryon_pipeline, _ensure_render,
_render_and_store) are removed.

What stays here is shared by outfits, jobs and the worker:
    cache identity (_single_key, _chain_key), cache lookup (_find_row),
    claiming / failing a cache row (_claim, _mark_failed),
    loading the avatar / a garment, FASHN category mapping.

Cache identity:
    (avatar_id, avatar_version, garment_id, asset_revision,
     provider, model, renderer_version)
A new avatar upload bumps avatar_version, an edited garment changes
asset_revision, a new model/settings bumps RENDERER_VERSION - each one
misses the cache and renders fresh. Old rows are simply never matched.
Chain steps use renderer_version "<v>-chain" and an asset_revision that
includes the first garment, so they never collide with single renders.

B8 (2026-11-23): the renderer now calls FASHN's own API instead of fal, but it is
the SAME model, so PROVIDER stays "fal" on purpose: it is part of the cache
identity, and changing it would make every existing render miss the cache
(users would pay again for looks they already have).

Rules the table design forces on us:
  * a 'processing' row older than STALE_AFTER is treated as dead
    (server crashed mid-render) and may be re-claimed;
  * a retry after 'failed' UPDATES the existing row, never inserts.
"""

import logging
from datetime import datetime, timedelta, timezone

from postgrest.exceptions import APIError

from core.storage.supabase_client import supabase
from shared.exceptions import (
    FitCheckException,
    AvatarNotFoundError,
    ClothingItemNotFoundError,
)

logger = logging.getLogger(__name__)

TABLE = "tryon_renders"

# Cache identity of what get_renderer() returns (see the B8 note above).
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


# ------------------------------------------------------------------ cache identity

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


# ------------------------------------------------------------------ claiming a cache row

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