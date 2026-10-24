"""
pipelines/avatar_delete_pipeline.py

AV-6: delete EVERYTHING derived from the user's photo (privacy).

  * avatar photos (original + processed) and any other files under the
    user's avatar folders in S3 (old versions, landmark/mask assets)
  * every try-on render of this user (tryon_renders rows + their S3 images)
  * every saved look (tryon_looks rows) - a look's picture IS a try-on render
  * the avatars row itself (body profile, pose keypoints, landmarks)

Order: S3 files first, then database rows. If a file delete fails it is
logged and the rows are still removed (nothing points at the file any more).
Blocking (boto3 + supabase) - call through run_in_threadpool.
"""

import logging

from core.storage.s3 import storage, S3Folders
from core.storage.supabase_client import supabase
from shared.exceptions import AvatarNotFoundError

logger = logging.getLogger(__name__)


def _key_from_url(url: str | None) -> str | None:
    if not url:
        return None
    return url.split(".amazonaws.com/")[-1] or None


def _delete_prefix(prefix: str) -> int:
    """Delete every S3 object under prefix. Returns how many were deleted."""
    deleted = 0
    try:
        paginator = storage.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=storage.bucket, Prefix=prefix):
            objects = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if not objects:
                continue
            storage.client.delete_objects(
                Bucket=storage.bucket,
                Delete={"Objects": objects, "Quiet": True},
            )
            deleted += len(objects)
    except Exception as exc:
        logger.warning("Avatar delete: could not clear S3 prefix %s: %s", prefix, exc)
    return deleted


def delete_avatar_everything(user_id: str) -> dict:
    avatar_res = supabase.table("avatars")\
        .select("avatar_id, original_photo_url, processed_photo_url")\
        .eq("user_id", user_id)\
        .maybe_single()\
        .execute()
    if not avatar_res or not avatar_res.data:
        raise AvatarNotFoundError()
    avatar = avatar_res.data

    renders = supabase.table("tryon_renders").select("id, s3_key")\
        .eq("user_id", user_id).execute().data or []
    looks = supabase.table("tryon_looks").select("look_id, tryon_key")\
        .eq("user_id", user_id).execute().data or []

    # 1. Files we know by key
    keys = {
        _key_from_url(avatar.get("original_photo_url")),
        _key_from_url(avatar.get("processed_photo_url")),
        *(r.get("s3_key") for r in renders),
        *(l.get("tryon_key") for l in looks),
    }
    keys.discard(None)

    files_deleted = 0
    for key in keys:
        if storage.delete(key):
            files_deleted += 1
        else:
            logger.warning("Avatar delete: S3 file not deleted: %s", key)

    # 2. Anything else left in the user's avatar folders (old versions, assets)
    for prefix in (f"{S3Folders.AVATARS}/{user_id}/", f"{S3Folders.AVATAR_ASSETS}/{user_id}/"):
        files_deleted += _delete_prefix(prefix)

    # 3. Database rows
    supabase.table("tryon_looks").delete().eq("user_id", user_id).execute()
    supabase.table("tryon_renders").delete().eq("user_id", user_id).execute()
    supabase.table("avatars").delete().eq("user_id", user_id).execute()

    result = {
        "deleted": True,
        "files_deleted": files_deleted,
        "renders_deleted": len(renders),
        "looks_deleted": len(looks),
    }
    logger.info("Avatar deleted - user=%s %s", user_id, result)
    return result