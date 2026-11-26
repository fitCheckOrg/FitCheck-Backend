"""
Try-On Studio B10 (2026-11-26): re-process existing avatars so processed_photo_url has
the same size + framing as the original photo (background removed, NOT cropped), and
recompute its body points. New uploads already do this (avatar_pipeline.py).

    python -m pipelines.avatar_reprocess --dry-run          # list, change nothing, $0
    python -m pipelines.avatar_reprocess --user <user_id>   # one avatar
    python -m pipelines.avatar_reprocess --all              # every avatar not done yet

* Costs 1 remove.bg credit per avatar actually re-processed.
* Skips an avatar whose processed photo already has the original's shape.
* avatar_version does NOT change: renders are made from the original photo,
  so caches, saved looks and outdated flags are untouched.
* If the avatar is replaced while this runs, the new file is deleted and nothing changes.
"""

import argparse
import io

from PIL import Image

from core.image.enhancer import enhance_image
from core.storage.s3 import storage, S3Folders
from core.storage.supabase_client import supabase
from tryon.vto_renderer import _get_with_retry
from workers.avatar.pose_estimator import estimate_pose
from workers.wardrobe.background import remove_background

RATIO_TOLERANCE = 0.01


def _key(url: str | None) -> str | None:
    return url.split(".amazonaws.com/")[-1] if url else None


def _uncropped(row: dict, original_size: tuple[int, int]) -> bool:
    w, h = row.get("photo_width"), row.get("photo_height")
    if not w or not h:
        return False
    return abs(w / h - original_size[0] / original_size[1]) <= RATIO_TOLERANCE


def reprocess(row: dict, dry_run: bool) -> str:
    uid = str(row["user_id"])
    original = _get_with_retry(row["original_photo_url"], "avatar original")
    o_size = Image.open(io.BytesIO(original)).size
    before = f"{row.get('photo_width')}x{row.get('photo_height')}"
    if _uncropped(row, o_size):
        print(f"  {uid}: already uncropped ({before}, original {o_size[0]}x{o_size[1]}) - skipped")
        return "skipped"
    if dry_run:
        print(f"  {uid}: would re-process (processed {before}, original {o_size[0]}x{o_size[1]})")
        return "would"

    processed = enhance_image(remove_background(original, crop=False))
    p_size = Image.open(io.BytesIO(processed)).size
    keypoints = estimate_pose(processed)
    up = storage.upload(processed, folder=f"{S3Folders.AVATARS}/{uid}/processed",
                        extension="png", content_type="image/png")
    try:
        res = supabase.table("avatars").update({
            "processed_photo_url": up["url"],
            "photo_width": p_size[0],
            "photo_height": p_size[1],
            "pose_keypoints": keypoints,
        }).eq("user_id", uid).eq("avatar_version", row["avatar_version"]).execute()
    except Exception:
        storage.delete(up["key"])
        raise
    if not res.data:
        storage.delete(up["key"])
        print(f"  {uid}: avatar changed while re-processing - left as it is")
        return "changed"

    old = _key(row.get("processed_photo_url"))
    if old and old != up["key"]:
        storage.delete(old)
    print(f"  {uid}: done - processed {before} -> {p_size[0]}x{p_size[1]} "
          f"(original {o_size[0]}x{o_size[1]}), body points {'found' if keypoints else 'NOT found'}")
    return "done"


def main():
    ap = argparse.ArgumentParser(description="Re-process avatars to the uncropped frame (B10).")
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--all", action="store_true")
    who.add_argument("--user")
    who.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    q = supabase.table("avatars").select(
        "user_id, avatar_version, original_photo_url, processed_photo_url, photo_width, photo_height")
    if args.user:
        q = q.eq("user_id", args.user)
    rows = q.execute().data or []
    print(f"{len(rows)} avatar(s)")
    counts: dict[str, int] = {}
    for row in rows:
        try:
            result = reprocess(row, dry_run=args.dry_run)
        except Exception as exc:
            print(f"  {row['user_id']}: FAILED - {exc}")
            result = "failed"
        counts[result] = counts.get(result, 0) + 1
    print("Summary:", counts)


if __name__ == "__main__":
    main()