from shared.logger import get_logger
from shared.exceptions import FitCheckException
from shared.models.avatar import AvatarProfile
from core.storage.supabase_client import supabase
import copy
import time

logger = get_logger(__name__)


def save_avatar(
    profile: AvatarProfile,
    body_profile_ai: dict,
    photo_width: int,
    photo_height: int,
    pose_keypoints: dict | None = None
) -> dict:
    """
    Upserts avatar to Supabase.
    One avatar per user — always.
    Returns saved row from database.
    """
    try:
        start = time.perf_counter()
        logger.info("Saving avatar — user=%s version=%d", 
                    profile.user_id, profile.avatar_version)

        data = {
            "user_id": str(profile.user_id),
            "original_photo_url": profile.original_photo_url,
            "processed_photo_url": profile.processed_photo_url,
            "photo_width": photo_width,
            "photo_height": photo_height,

            # AI result — stored separately, never overwritten
            "body_profile_ai": copy.deepcopy(body_profile_ai),


            # Active profile — what the app uses
            "body_profile": profile.body.model_dump(mode="json"),


            "avatar_version": profile.avatar_version,
            "is_setup": profile.is_setup,
            "status": "ready",
            
            "pose_keypoints": pose_keypoints,
        }

        # Upsert — insert or update on user_id conflict
        result = supabase.table("avatars")\
            .upsert(data, on_conflict="user_id")\
            .execute()

        if not result.data:
            raise FitCheckException(
                "Failed to save avatar.",
                code="DATABASE_ERROR",
                status_code=500
            )

        elapsed = time.perf_counter() - start
        logger.info(
            "Avatar saved in %.2fs user=%s version=%d",
            elapsed,
            profile.user_id,
            profile.avatar_version
        )
        return result.data[0]


    except FitCheckException:
        raise
    except Exception:
        logger.exception("Failed to save avatar")
        raise FitCheckException(
            "Something went wrong saving your avatar. Please try again.",
            code="DATABASE_ERROR",
            status_code=500
        )