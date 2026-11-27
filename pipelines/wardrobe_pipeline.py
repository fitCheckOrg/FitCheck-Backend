from fastapi import UploadFile
from shared.logger import get_logger
from shared.models.clothing import ClothingItem
from core.image.validator import validate_image, check_free_tier_limit
from core.storage.s3 import storage, S3Folders
from core.storage.supabase_client import supabase
from workers.wardrobe.background import remove_background
from workers.wardrobe.classifier import classify_clothing
from workers.wardrobe.cutout import make_cutout
from workers.wardrobe.metadata import build_clothing_item
from workers.wardrobe.saver import save_clothing_item

logger = get_logger(__name__)

CUTOUT_VERSION = 1


async def run_wardrobe_pipeline(
    file: UploadFile,
    user_id: str
) -> ClothingItem:
    """
    Full wardrobe processing pipeline.

    Upload -> Validate -> Remove BG (uncropped) -> Sharp cut-out ->
    Classify -> Build Item -> Save to DB -> Return

    Try-On Studio B14-0 (2026-11-27):
      * the original upload is kept untouched (original_image_url) - always;
      * remove.bg runs WITHOUT cropping and only its outline is used: the cut-out is the
        ORIGINAL photo's pixels (up to MAX_IMAGE_SIZE) trimmed to the garment + 3 %
        (workers/wardrobe/cutout.py) - sharp instead of remove.bg's ~0.25 MP preview;
      * no colour / contrast / sharpness "enhancement" any more: the garment keeps its
        real colours on the board and in renders;
      * the same file is clean_image_url (render input) and cutout_url (board);
        cutout_width / cutout_height / cutout_bbox describe it.

    Max execution time: 30 seconds
    Retries: handled inside each worker
    Rollback: S3 cleanup on failure
    """

    original_key = None
    clean_key = None

    try:
        logger.info(f"Wardrobe pipeline started - user: {user_id}")

        # 0. Free tier check - before any expensive operations
        check_free_tier_limit(user_id)
        logger.info("Step 0/7 - Free tier check passed")

        # 1. Read image bytes
        image_bytes = await file.read()
        content_type = file.content_type

        # 2. Validate
        validate_image(image_bytes, content_type)
        logger.info("Step 1/7 - Validation passed")

        # 3. Upload original to S3 (kept permanently)
        original_result = storage.upload(
            image_bytes,
            folder=f"{S3Folders.WARDROBE}/{user_id}/original",
            extension="png",
            content_type=content_type
        )
        original_key = original_result["key"]
        logger.info("Step 2/7 - Original uploaded to S3")

        # 4. Remove background - uncropped, so its outline fits the original (B14-0)
        removebg_bytes = remove_background(image_bytes, crop=False)
        logger.info("Step 3/7 - Background removed")

        # 5. Sharp cut-out: original pixels + remove.bg's outline, trimmed (B14-0)
        cut = make_cutout(image_bytes, removebg_bytes)
        cut_bytes = cut["bytes"]
        logger.info("Step 4/7 - Cut-out %dx%d (%s)", cut["width"], cut["height"], cut["source"])

        # 6. Upload the cut-out to S3
        clean_result = storage.upload(
            cut_bytes,
            folder=f"{S3Folders.WARDROBE}/{user_id}/clean",
            extension="png",
            content_type="image/png"
        )
        clean_key = clean_result["key"]
        logger.info("Step 5/7 - Cut-out uploaded to S3")

        # 7. Classify with GPT-4o Vision
        classification = classify_clothing(cut_bytes, "image/png")
        logger.info("Step 6/7 - Classification complete")

        # 8. Build final ClothingItem
        item = build_clothing_item(
            user_id=user_id,
            classification=classification,
            original_image_url=original_result["url"],
            clean_image_url=clean_result["url"],
            image_bytes=cut_bytes
        )

        # 9. Save to Supabase, then the B14-0 cut-out fields
        save_clothing_item(item, cut_bytes)
        _save_cutout_fields(str(item.item_id), clean_result["url"], cut)
        item.cutout_url = clean_result["url"]
        item.cutout_width, item.cutout_height = cut["width"], cut["height"]
        item.cutout_bbox = cut["bbox"]
        logger.info("Step 7/7 - Saved to database")

        logger.info(
            f"Wardrobe pipeline complete - "
            f"{item.category} / {item.item_type} / "
            f"confidence: {item.classification_confidence}"
        )

        return item

    except Exception as e:
        logger.error(f"Wardrobe pipeline failed: {e}")
        _rollback(original_key, clean_key)
        raise


def _save_cutout_fields(item_id: str, url: str, cut: dict) -> None:
    """B14-0 fields on the new item. Never fails the upload (the item is already saved)."""
    from datetime import datetime, timezone
    try:
        supabase.table("closet_items").update({
            "cutout_url": url,
            "cutout_width": cut["width"],
            "cutout_height": cut["height"],
            "cutout_bbox": cut["bbox"],
            "cutout_version": CUTOUT_VERSION,
            "cutout_status": "done",
            "cutout_error": None,
            "cutout_at": datetime.now(timezone.utc).isoformat(),
        }).eq("item_id", item_id).execute()
    except Exception:
        logger.warning("Saving cut-out fields failed - item=%s", item_id)


def _rollback(original_key: str | None, clean_key: str | None) -> None:
    """
    Clean up S3 uploads if pipeline fails.
    Prevents orphaned files in storage.
    """
    if original_key:
        deleted = storage.delete(original_key)
        logger.info(f"Rollback - original deleted: {deleted}")

    if clean_key:
        deleted = storage.delete(clean_key)
        logger.info(f"Rollback - clean image deleted: {deleted}")