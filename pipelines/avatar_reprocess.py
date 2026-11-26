"""
Try-On Studio B10 / B10b (2026-11-26): bring existing avatars up to date:
  * stage photo (processed_photo_url) = the original photo's pixels, up to
    MAX_IMAGE_SIZE, with the cut-out's transparency - same frame as the original;
  * body points (pose_keypoints) and outline points (body_outline) recomputed on it.
New uploads already do this (avatar_pipeline.py).

    python -m pipelines.avatar_reprocess --dry-run          # list, change nothing, $0
    python -m pipelines.avatar_reprocess --user <user_id>   # one avatar
    python -m pipelines.avatar_reprocess --all              # every avatar not done yet

* $0 for avatars already uncropped (B10): the current cut-out's transparency is reused.
  1 remove.bg credit only for an avatar still on an old cropped cut-out.
* Skips an avatar that is already up to date.
* avatar_version does NOT change: renders are made from the original photo,
  so caches, saved looks and outdated flags are untouched.
* If the avatar is replaced while this runs, the new file is deleted and nothing changes.
"""

import argparse
import io

from PIL import Image, ImageOps

from core.config.settings import settings
from core.storage.s3 import storage, S3Folders
from core.storage.supabase_client import supabase
from tryon.vto_renderer import _get_with_retry
from workers.avatar.stage_photo import make_stage
from workers.wardrobe.background import remove_background

RATIO_TOLERANCE = 0.01


def _key(url: str | None) -> str | None:
    return url.split(".amazonaws.com/")[-1] if url else None


def _target_size(original: bytes) -> tuple[int, int]:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(original)))
    img.thumbnail((int(settings.MAX_IMAGE_SIZE),) * 2)
    return img.size


def reprocess(row: dict, dry_run: bool) -> str:
    uid = str(row["user_id"])
    original = _get_with_retry(row["original_photo_url"], "avatar original")
    target = _target_size(original)
    w, h = row.get("photo_width") or 0, row.get("photo_height") or 0
    before = f"{w}x{h}"

    if (w, h) == target and row.get("body_outline"):
        print(f"  {uid}: up to date ({before}, outline stored) - skipped")
        return "skipped"

    uncropped = bool(w and h) and abs(w / h - target[0] / target[1]) <= RATIO_TOLERANCE
    cost = "$0, reuses the cut-out" if uncropped else "1 remove.bg credit"
    if dry_run:
        print(f"  {uid}: would update ({before} -> {target[0]}x{target[1]}, {cost})")
        return "would"

    if uncropped:
        cutout = _get_with_retry(row["processed_photo_url"], "avatar processed")
    else:
        cutout = remove_background(original, crop=False)

    stage = make_stage(original, cutout)
    up = storage.upload(stage["bytes"], folder=f"{S3Folders.AVATARS}/{uid}/processed",
                        extension="png", content_type="image/png")
    try:
        res = supabase.table("avatars").update({
            "processed_photo_url": up["url"],
            "photo_width": stage["width"],
            "photo_height": stage["height"],
            "pose_keypoints": stage["pose_keypoints"],
            "body_outline": stage["body_outline"],
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
    outline = stage["body_outline"]
    print(f"  {uid}: done - {before} -> {stage['width']}x{stage['height']} ({cost}), "
          f"body points {'found' if stage['pose_keypoints'] else 'NOT found'}, "
          f"outline {str(len(outline)) + ' points' if outline else 'NOT measured'}")
    return "done"


def main():
    ap = argparse.ArgumentParser(description="Bring avatars up to date (B10 / B10b).")
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--all", action="store_true")
    who.add_argument("--user")
    who.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    q = supabase.table("avatars").select(
        "user_id, avatar_version, original_photo_url, processed_photo_url, "
        "photo_width, photo_height, body_outline")
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