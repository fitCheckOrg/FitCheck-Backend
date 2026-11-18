"""
Try-On v2 T2-1 B6 (2026-11-18): out-of-date renders and their cleanup.

    Garment deleted   -> every render showing it: outdated "garment_deleted"
    Avatar replaced   -> every render of an older avatar version: outdated "avatar_replaced"

    * The render's file is deleted from S3 (file_deleted_at set) - EXCEPT when a
      saved look uses it: looks keep their image, marked outdated as information.
    * Outfits keep the piece with missing: true (B1). Jobs for an old avatar
      version finish as failed / avatar_missing and aren't charged (B2 worker).
    * Avatar deleted (AV-6) still removes everything, looks included.

In API image objects: {..., "outdated": "garment_deleted" | "avatar_replaced",
"url": null when the file is gone}. Best effort: a cleanup problem is logged and
never fails the delete / upload that triggered it.

B6 fix: a file whose delete failed (connection drop) is retried by sweep_files(),
run by cleanup_worker() every SWEEP_EVERY seconds (started in main.py).
"""

import asyncio
import logging
from datetime import datetime, timezone

from core.storage.s3 import storage
from core.storage.supabase_client import supabase

logger = logging.getLogger(__name__)

SWEEP_EVERY = 600          # seconds between cleanup passes
SWEEP_BATCH = 100          # files per pass


def _mark_deleted(ids: list) -> None:
    if ids:
        supabase.table("tryon_renders").update({"file_deleted_at": datetime.now(timezone.utc).isoformat()})\
            .in_("id", ids).execute()


def _run(user_id: str, reason: str, garment_id: str | None = None, version: int | None = None) -> dict:
    out = {"outdated": 0, "files_deleted": 0, "kept_by_looks": 0}
    try:
        rows = supabase.rpc("outdate_renders", {
            "p_user": str(user_id), "p_reason": reason,
            "p_garment": str(garment_id) if garment_id else None,
            "p_version": version,
        }).execute().data or []
        out["outdated"] = len(rows)
        gone = []
        for r in rows:
            if r.get("kept"):
                out["kept_by_looks"] += 1
            elif r.get("render_key") and storage.delete(r["render_key"]):
                gone.append(r["render_id"])
        _mark_deleted(gone)
        out["files_deleted"] = len(gone)
        logger.info("renders outdated (%s) user=%s %s", reason, user_id, out)
    except Exception:
        logger.exception("outdating renders failed (%s) user=%s", reason, user_id)
    return out


def outdate_garment(user_id, garment_id) -> dict:
    """After a garment is deleted."""
    return _run(str(user_id), "garment_deleted", garment_id=str(garment_id))


def outdate_avatar(user_id, new_version: int) -> dict:
    """After the avatar was replaced with version new_version."""
    return _run(str(user_id), "avatar_replaced", version=int(new_version))


def sweep_files(limit: int = SWEEP_BATCH) -> int:
    """Delete files of outdated renders that are still there (an earlier delete failed).
    Never a file a saved look keeps. Returns how many were deleted."""
    rows = supabase.rpc("outdated_files_to_delete", {"p_limit": limit}).execute().data or []
    gone = [r["render_id"] for r in rows if storage.delete(r["render_key"])]
    _mark_deleted(gone)
    if rows:
        logger.info("cleanup: %s of %s outdated render files deleted", len(gone), len(rows))
    return len(gone)


async def cleanup_worker() -> None:
    """Background loop started with the app."""
    logger.info("cleanup worker started (every %ss)", SWEEP_EVERY)
    await asyncio.sleep(60)
    while True:
        try:
            await asyncio.to_thread(sweep_files)
        except Exception:
            logger.exception("cleanup pass failed")
        await asyncio.sleep(SWEEP_EVERY)