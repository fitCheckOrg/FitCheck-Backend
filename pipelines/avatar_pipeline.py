from fastapi import UploadFile
from shared.logger import get_logger
from shared.models.avatar import AvatarProfile
from core.image.validator import validate_image
from core.image.enhancer import enhance_image
from core.storage.s3 import storage, S3Folders
from core.storage.supabase_client import supabase
from workers.wardrobe.background import remove_background
from workers.avatar.analyzer import analyze_body
from workers.avatar.generator import build_avatar_profile
from workers.avatar.saver import save_avatar
from workers.avatar.pose_estimator import estimate_pose

logger = get_logger(__name__)


async def run_avatar_pipeline(
    file: UploadFile,
    user_id: str
) -> AvatarProfile:
    """
    Full avatar processing pipeline.

    Validate → Remove BG → Enhance → Analyze →
    Upload Images → Build Profile → Save → Cleanup old

    One avatar per user — upserts on conflict.
    Rollback: deletes new S3 images if save fails.
    """

    original_key: str | None = None
    processed_key: str | None = None
    old_original_url: str | None = None
    old_processed_url: str | None = None

    try:
        logger.info("Avatar pipeline started — user=%s", user_id)

        # 1. Read image
        image_bytes = await file.read()
        content_type = file.content_type

        # 2. Validate — returns PIL Image so we don't decode twice
        validated_image = validate_image(image_bytes, content_type)
        photo_width, photo_height = validated_image.size
        logger.info("Step 1/7 — Validation passed %dx%d", photo_width, photo_height)

        # 3. Get existing avatar — version + old S3 URLs for cleanup
        existing_version, old_original_url, old_processed_url = _get_existing_avatar(user_id)

        # 4. Remove background
        processed_bytes = remove_background(image_bytes)
        logger.info("Step 2/7 — Background removed")

        # 5. Enhance
        enhanced_bytes = enhance_image(processed_bytes)
        logger.info("Step 3/7 — Image enhanced")

        # Capture ACTUAL final dimensions — after crop, after enhance —
        # not the original pre-crop dimensions. remove.bg's crop=true
        # (added for wardrobe, shared by this pipeline) changes both
        # size AND aspect ratio, so this can't be assumed to match
        # the original upload.
        from PIL import Image
        import io as _io
        final_image = Image.open(_io.BytesIO(enhanced_bytes))
        photo_width, photo_height = final_image.size
        logger.info("Actual processed dimensions: %dx%d", photo_width, photo_height)

        # 6. Analyze body BEFORE uploading — avoid S3 cost if GPT fails
        body_profile, body_profile_ai, orientation, accuracy = analyze_body(enhanced_bytes, "image/png")
        body_profile_ai["detected_orientation"] = orientation
        body_profile_ai["photo_accuracy"] = accuracy
        logger.info("Step 4/7 — Body analysis complete")

        # 4b. Estimate pose keypoints on the FINAL PROCESSED photo —
        # must match the coordinate system of processed_photo_url,
        # which is what pose_keypoints will be applied against
        # downstream. Previously ran on image_bytes (the ORIGINAL,
        # pre-crop upload) while processed_photo_url stores the
        # POST-crop image — a real aspect-ratio mismatch (confirmed
        # via direct measurement: 0.5627 vs 0.3314, 41.1% difference)
        # that silently corrupted every downstream landmark
        # calculation using these normalized coordinates.
        pose_keypoints = estimate_pose(enhanced_bytes)
        if pose_keypoints:
            logger.info("Step 4b/7 — Pose keypoints detected")
        else:
            logger.warning("Step 4b/7 — Pose keypoints not detected, continuing without them")

        # 7. Upload original to S3
        original_result = storage.upload(
            image_bytes,
            folder=f"{S3Folders.AVATARS}/{user_id}/original",
            extension="png",
            content_type=content_type
        )
        original_key = original_result["key"]
        logger.info("Step 5/7 — Original uploaded to S3")

        # 8. Upload processed to S3
        processed_result = storage.upload(
            enhanced_bytes,
            folder=f"{S3Folders.AVATARS}/{user_id}/processed",
            extension="png",
            content_type="image/png"
        )
        processed_key = processed_result["key"]
        logger.info("Step 6/7 — Processed uploaded to S3")

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
        logger.info("Step 7/7 — Avatar saved to database")

        # 11. Delete old S3 images after successful save
        _cleanup_old_images(old_original_url, old_processed_url)

        logger.info(
            "Avatar pipeline complete — user=%s version=%d shape=%s",
            user_id,
            profile.avatar_version,
            body_profile.body_shape.value
        )

        return saved_avatar

    except Exception as e:
        logger.exception("Avatar pipeline failed")
        _rollback(original_key, processed_key)
        raise


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
        logger.info("Rollback — original deleted: %s", deleted)

    if processed_key:
        deleted = storage.delete(processed_key)
        logger.info("Rollback — processed deleted: %s", deleted)