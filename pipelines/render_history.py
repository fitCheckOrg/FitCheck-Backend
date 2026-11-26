"""
Try-On Studio B12 (2026-11-26): recent renders - the Studio's History (S2, for ST10).

GET /api/tryon/jobs?status=succeeded&limit=30&offset=0
  -> {"jobs": [{job_id, outfit_id, created_at, finished_at,
                items: [{slot, garment_id}],
                image: {image_id, key, url, expires_at, outdated} | null}],
      "limit", "offset", "has_more"}

* Newest first. Each render appears ONCE, at the latest time it was made or opened
  (opening an already-rendered outfit records a free "succeeded" job; those are folded
  into the same entry instead of repeating the photo).
* Outdated renders are included, with `image.outdated` set ("avatar_replaced" /
  "garment_deleted"); `url` is null when the file was deleted.
* items = every piece of the outfit (tray pieces too), so tapping one can wear the
  whole outfit again - free, the render is cached.
* Only the caller's own renders (list_render_history runs for one user and can't be
  called by app users directly).
"""

import logging

from core.storage.supabase_client import supabase
from pipelines.render_jobs_pipeline import _bad

logger = logging.getLogger(__name__)

DEFAULT_LIMIT = 30
MAX_LIMIT = 50


def _outfit_items(outfit_ids: list[str]) -> dict[str, list[dict]]:
    if not outfit_ids:
        return {}
    try:
        rows = supabase.table("tryon_outfit_items").select("outfit_id, slot, position, garment_id")\
            .in_("outfit_id", sorted(set(outfit_ids))).execute().data or []
    except Exception:
        logger.warning("render history: outfit items not readable, using the rendered pieces")
        return {}
    out: dict[str, list[dict]] = {}
    for r in sorted(rows, key=lambda r: (str(r["outfit_id"]), int(r.get("position") or 0))):
        out.setdefault(str(r["outfit_id"]), []).append(
            {"slot": r["slot"], "garment_id": str(r["garment_id"])})
    return out


def list_history(user_id, limit: int = DEFAULT_LIMIT, offset: int = 0) -> dict:
    from pipelines.tryon_outfits_pipeline import _image
    uid = str(user_id)
    if not (1 <= int(limit) <= MAX_LIMIT) or int(offset) < 0:
        raise _bad("PAGE_INVALID", f"limit must be 1-{MAX_LIMIT} and offset 0 or more.")
    limit, offset = int(limit), int(offset)

    rows = supabase.rpc("list_render_history", {
        "p_user": uid, "p_limit": limit + 1, "p_offset": offset,
    }).execute().data or []
    has_more = len(rows) > limit
    rows = rows[:limit]

    image_ids = [str(r["image_id"]) for r in rows]
    renders: dict[str, dict] = {}
    if image_ids:
        for r in supabase.table("tryon_renders").select("id, s3_key, status, outdated, file_deleted_at")\
                .in_("id", image_ids).execute().data or []:
            renders[str(r["id"])] = r
    items_by_outfit = _outfit_items([str(r["outfit_id"]) for r in rows])

    jobs = []
    for r in rows:
        steps = r.get("steps") or []
        render = renders.get(str(r["image_id"]))
        if render and render.get("s3_key"):
            image = _image(render, {"avatar_version": r["avatar_version"]},
                           [s["garment_id"] for s in steps])
        elif render:
            image = {"image_id": str(render["id"]), "key": None, "url": None,
                     "expires_at": None, "outdated": render.get("outdated")}
        else:
            image = None
        jobs.append({
            "job_id": str(r["job_id"]),
            "outfit_id": str(r["outfit_id"]),
            "created_at": r["created_at"],
            "finished_at": r.get("finished_at"),
            "items": items_by_outfit.get(str(r["outfit_id"]))
            or [{"slot": s["slot"], "garment_id": str(s["garment_id"])} for s in steps],
            "image": image,
        })
    return {"jobs": jobs, "limit": limit, "offset": offset, "has_more": has_more}