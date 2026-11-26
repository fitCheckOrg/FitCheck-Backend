from fastapi import UploadFile
from shared.logger import get_logger
from shared.models.avatar import AvatarProfile
from core.image.validator import validate_image
from core.storage.s3 import storage, S3Folders
from core.storage.supabase_client import supabase
from workers.wardrobe.background import remove_background
from workers.avatar.analyzer import analyze_body
from workers.avatar.generator import build_avatar_profile
from workers.avatar.saver import save_avatar
from workers.avatar.stage_photo import make_stage

logger = get_logger(__name__)


async def run_avatar_pipeline(
    file: UploadFile,
    user_id: str
) -> AvatarProfile:
    """
    Full avatar processing pipeline.

    Validate -> Remove BG -> Stage photo -> Pose + Outline -> Analyze ->
    Upload Images -> Build Profile -> Save -> Cleanup old -> Outdate old renders

    Try-On v2 B7 (2026-11-18): the underarm check (old mesh try-on) is gone.

    Try-On Studio B10 (2026-11-26): the background is removed WITHOUT cropping, so
    processed_photo_url has the same framing as the original photo - which is what
    FASHN renders from. Stage photo, body points and renders share one picture frame.

    Try-On Studio B10b (2026-11-26):
      * the stage photo is the ORIGINAL photo's pixels (up to MAX_IMAGE_SIZE) with
        remove.bg's transparency on top - remove.bg's own result is only ~0.25 MP;
      * no colour/contrast/brightness "enhancement" on avatars any more: the person's
        skin and face stay exactly as photographed, and match the renders;
      * body_outline: neck / shoulder / waist / hip edges measured from the cut-out.

    One avatar per user - upserts on conflict.
    Rollback: deletes new S3 images if save fails.
    """

    original_key: str | None = None
    processed_key: str | None = None
    old_original_url: str | None = None
    old_processed_url: str | None = None

    try:
        logger.info("Avatar pipeline started - user=%s", user_id)

        # 1. Read image
        image_bytes = await file.read()
        content_type = file.content_type

        # 2. Validate - returns PIL Image so we don't decode twice
        validated_image = validate_image(image_bytes, content_type)
        logger.info("Step 1/7 - Validation passed %dx%d", *validated_image.size)

        # 3. Get existing avatar - version + old S3 URLs for cleanup
        existing_version, old_original_url, old_processed_url = _get_existing_avatar(user_id)

        # 4. Remove background - no crop (B10): same framing as the original
        cutout_bytes = remove_background(image_bytes, crop=False)
        logger.info("Step 2/7 - Background removed (uncropped)")

        # 5. Stage photo + body points + outline - one picture frame (B10b)
        stage = make_stage(image_bytes, cutout_bytes)
        stage_bytes = stage["bytes"]
        photo_width, photo_height = stage["width"], stage["height"]
        pose_keypoints = stage["pose_keypoints"]
        body_outline = stage["body_outline"]
        logger.info("Step 3/7 - Stage photo %dx%d, body points %s, outline %s",
                    photo_width, photo_height,
                    "found" if pose_keypoints else "NOT found",
                    f"{len(body_outline)} points" if body_outline else "NOT measured")

        # 6. Analyze body BEFORE uploading - avoid S3 cost if GPT fails
        body_profile, body_profile_ai, orientation, accuracy = analyze_body(stage_bytes, "image/png")
        body_profile_ai["detected_orientation"] = orientation
        body_profile_ai["photo_accuracy"] = accuracy
        logger.info("Step 4/7 - Body analysis complete")

        # 7. Upload original to S3
        original_result = storage.upload(
            image_bytes,
            folder=f"{S3Folders.AVATARS}/{user_id}/original",
            extension="png",
            content_type=content_type
        )
        original_key = original_result["key"]
        logger.info("Step 5/7 - Original uploaded to S3")

        # 8. Upload stage photo to S3
        processed_result = storage.upload(
            stage_bytes,
            folder=f"{S3Folders.AVATARS}/{user_id}/processed",
            extension="png",
            content_type="image/png"
        )
        processed_key = processed_result["key"]
        logger.info("Step 6/7 - Stage photo uploaded to S3")

        # 9. Build AvatarProfile
        profile = build_avatar_profile(
            user_id=user_id,
            body_profile=body_profile,
            original_photo_url=original_result["url"],
            processed_photo_url=processed_result["url"],
            avatar_version=existing_version + 1
        )

        # 10. Save to Supabase
        saved_avatar = save_avatar(
            profile=profile,
            body_profile_ai=body_profile_ai,
            photo_width=photo_width,
            photo_height=photo_height,
            pose_keypoints=pose_keypoints
        )
        saved_avatar = _as_dict(saved_avatar)
        saved_avatar["body_outline"] = _save_outline(user_id, profile.avatar_version, body_outline)
        logger.info("Step 7/7 - Avatar saved to database")

        # 11. Delete old S3 images after successful save
        _cleanup_old_images(old_original_url, old_processed_url)

        # 12. Try-On v2 B6: renders of the old avatar are outdated; their files are
        # deleted unless a saved look keeps them. Never fails the upload.
        if profile.avatar_version > 1:
            from pipelines.outdated_pipeline import outdate_avatar
            outdate_avatar(user_id, profile.avatar_version)

        logger.info(
            "Avatar pipeline complete - user=%s version=%d shape=%s",
            user_id,
            profile.avatar_version,
            body_profile.body_shape.value
        )

        return saved_avatar

    except Exception:
        logger.exception("Avatar pipeline failed")
        _rollback(original_key, processed_key)
        raise


def _as_dict(saved) -> dict:
    if isinstance(saved, dict):
        return dict(saved)
    if hasattr(saved, "model_dump"):
        return saved.model_dump()
    return dict(saved)


def _save_outline(user_id: str, avatar_version: int, body_outline: dict | None) -> dict | None:
    """Stores body_outline on the avatar just saved. Never fails the upload."""
    if not body_outline:
        return None
    try:
        supabase.table("avatars")\
            .update({"body_outline": body_outline})\
            .eq("user_id", user_id)\
            .eq("avatar_version", avatar_version)\
            .execute()
        return body_outline
    except Exception:
        logger.warning("Saving body_outline failed - user=%s", user_id)
        return None


def _get_existing_avatar(user_id: str) -> tuple[int, str | None, str | None]:
    """
    Get existing avatar version and S3 URLs.
    Returns (version, original_url, processed_url).
    Returns (0, None, None) if no avatar exists.
    """
    try:
        result = supabase.table("avatars")\
            .select("avatar_version, original_photo_url, processed_photo_url")\
            .eq("user_id", user_id)\
            .single()\
            .execute()

        if result.data:
            return (
                result.data.get("avatar_version", 0),
                result.data.get("original_photo_url"),
                result.data.get("processed_photo_url")
            )
        return 0, None, None
    except Exception:
        return 0, None, None


def _cleanup_old_images(
    old_original_url: str | None,
    old_processed_url: str | None
) -> None:
    """Delete old S3 images after successful avatar replacement."""
    for url in [old_original_url, old_processed_url]:
        if url:
            try:
                key = url.split(".amazonaws.com/")[-1]
                storage.delete(key)
                logger.info("Old avatar image deleted: %s", key)
            except Exception:
                logger.warning("Failed to delete old avatar image: %s", url)


def _rollback(
    original_key: str | None,
    processed_key: str | None
) -> None:
    """Delete uploaded S3 images if pipeline fails."""
    if original_key:
        deleted = storage.delete(original_key)
        logger.info("Rollback - original deleted: %s", deleted)

    if processed_key:
        deleted = storage.delete(processed_key)
        logger.info("Rollback - processed deleted: %s", deleted)