"""
Try-On Studio B14-0 (2026-11-27): sharp cut-outs for garments that already exist.

    python -m pipelines.wardrobe_cutout_reprocess --dry-run            # count + remove.bg allowance, $0
    python -m pipelines.wardrobe_cutout_reprocess --item <item_id>     # one garment
    python -m pipelines.wardrobe_cutout_reprocess --user <user_id> --limit 10
    python -m pipelines.wardrobe_cutout_reprocess --all --limit 10     # next 10 not done yet
    add --retry-failed to try garments that failed before

* Uses 1 remove.bg call per garment (free preview size). --limit (default 10) caps a batch;
  the run stops at once when remove.bg says the allowance is used up (HTTP 402 / 429).
* Safe to stop and run again: a garment is done once (cutout_status = 'done') and is
  never processed twice; failures are recorded (cutout_status = 'failed', cutout_error)
  and skipped unless --retry-failed.
* Never touches original_image_url (kept permanently), clean_image_url or image_hash:
  renders and their caches stay exactly as they are. The sharp file only fills
  cutout_url / cutout_width / cutout_height / cutout_bbox (the board's image).
"""

import argparse
from datetime import datetime, timezone

import requests

from core.config.settings import settings
from core.storage.s3 import storage, S3Folders
from core.storage.supabase_client import supabase
from shared.exceptions import BackgroundRemovalError
from tryon.vto_renderer import _get_with_retry
from workers.wardrobe.background import remove_background
from workers.wardrobe.cutout import make_cutout, CutoutError

CUTOUT_VERSION = 1
FIELDS = "item_id, user_id, original_image_url, category, item_type, cutout_status"


class _Stop(Exception):
    """remove.bg refuses (allowance used up / rate limit): stop the whole batch."""


def removebg_allowance() -> str:
    """remove.bg's own account numbers (no key or account details printed)."""
    try:
        r = requests.get("https://api.remove.bg/v1.0/account",
                         headers={"X-Api-Key": settings.REMOVE_BG_API_KEY}, timeout=20)
        if r.status_code != 200:
            return f"couldn't read (HTTP {r.status_code})"
        a = (r.json().get("data") or {}).get("attributes") or {}
        free = (a.get("api") or {}).get("free_calls")
        credits = (a.get("credits") or {}).get("total")
        return f"free preview calls left: {free}, paid credits: {credits}"
    except Exception as exc:
        return f"couldn't read ({type(exc).__name__})"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mark(item_id: str, fields: dict) -> None:
    supabase.table("closet_items").update({**fields, "cutout_at": _now()})\
        .eq("item_id", item_id).execute()


def process(row: dict) -> str:
    iid, uid = str(row["item_id"]), str(row["user_id"])
    label = f"{iid[:8]} {row.get('category')}/{row.get('item_type')}"
    try:
        original = _get_with_retry(row["original_image_url"], "garment original")
    except Exception as exc:
        _mark(iid, {"cutout_status": "failed", "cutout_error": f"original: {exc}"[:300]})
        print(f"  {label}: FAILED - original not readable")
        return "failed"
    try:
        removebg = remove_background(original, crop=False)
    except BackgroundRemovalError as exc:
        raise _Stop(str(exc))
    try:
        cut = make_cutout(original, removebg)
    except CutoutError as exc:
        _mark(iid, {"cutout_status": "failed", "cutout_error": str(exc)[:300]})
        print(f"  {label}: FAILED - {exc}")
        return "failed"

    up = storage.upload(cut["bytes"], folder=f"{S3Folders.WARDROBE}/{uid}/clean",
                        extension="png", content_type="image/png")
    try:
        res = supabase.table("closet_items").update({
            "cutout_url": up["url"], "cutout_width": cut["width"], "cutout_height": cut["height"],
            "cutout_bbox": cut["bbox"], "cutout_version": CUTOUT_VERSION,
            "cutout_status": "done", "cutout_error": None, "cutout_at": _now(),
        }).eq("item_id", iid).is_("cutout_url", "null").execute()
    except Exception:
        storage.delete(up["key"])
        raise
    if not res.data:                       # done by another run meanwhile, or deleted
        storage.delete(up["key"])
        print(f"  {label}: already done / gone - skipped")
        return "skipped"
    print(f"  {label}: done - {cut['width']}x{cut['height']} ({cut['source']})")
    return "done"


def main():
    ap = argparse.ArgumentParser(description="Sharp garment cut-outs for existing clothes (B14-0).")
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--dry-run", action="store_true")
    who.add_argument("--item")
    who.add_argument("--user")
    who.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()

    q = supabase.table("closet_items").select(FIELDS).is_("cutout_url", "null")
    if args.item:
        q = q.eq("item_id", args.item)
    if args.user:
        q = q.eq("user_id", args.user)
    rows = q.order("created_at").execute().data or []
    failed = [r for r in rows if r.get("cutout_status") == "failed"]
    todo = rows if args.retry_failed else [r for r in rows if r.get("cutout_status") != "failed"]

    print(f"Not done yet: {len(rows)} (never tried {len(rows) - len(failed)}, failed before {len(failed)})")
    print(f"remove.bg: {removebg_allowance()}")
    if args.dry_run:
        for r in todo[:50]:
            print(f"  would do {str(r['item_id'])[:8]} {r.get('category')}/{r.get('item_type')}")
        print(f"Would use {len(todo)} remove.bg call(s); a batch is capped at --limit {args.limit}.")
        return

    batch = todo[: max(1, args.limit)]
    print(f"This batch: {len(batch)}")
    counts: dict[str, int] = {}
    for row in batch:
        try:
            result = process(row)
        except _Stop as exc:
            print(f"  STOPPED - remove.bg refused: {exc}. Nothing else was sent; run again later.")
            counts["stopped"] = 1
            break
        except Exception as exc:
            _mark(str(row["item_id"]), {"cutout_status": "failed",
                                         "cutout_error": f"{type(exc).__name__}: {exc}"[:300]})
            print(f"  {str(row['item_id'])[:8]}: FAILED - {type(exc).__name__}")
            result = "failed"
        counts[result] = counts.get(result, 0) + 1
    print("Summary:", counts)
    print(f"Still to do after this batch: {len(todo) - counts.get('done', 0) - counts.get('skipped', 0)}")


if __name__ == "__main__":
    main()