from fastapi import APIRouter, UploadFile, File, Form, Query, Depends
from pydantic import BaseModel
from uuid import UUID
from shared.logger import get_logger
from shared.models.responses import SuccessResponse
from shared.exceptions import AvatarNotFoundError
from core.storage.supabase_client import supabase
from core.storage.s3 import storage
from pipelines.avatar_pipeline import run_avatar_pipeline
from shared.dependencies import get_current_user_id
from shared.models.avatar_asset import AvatarAsset
from workers.avatar.asset_builder import build_avatar_asset


logger = get_logger(__name__)

router = APIRouter(prefix="/avatar", tags=["Avatar"])


class UpdateAvatarRequest(BaseModel):
    display_height: str | None = None
    display_body_shape: str | None = None
    display_skin_tone: str | None = None
    hair_style: str | None = None
    hair_color: str | None = None


@router.get("/health")
async def health():
    return { "service": "avatar", "version": "1.0.0", "status": "healthy" }


@router.post("/upload", response_model=SuccessResponse)
async def upload_avatar(
    file: UploadFile = File(...),
    user_id: UUID = Depends(get_current_user_id)
):
    logger.info(f"Avatar upload — user={user_id}")
    saved_avatar = await run_avatar_pipeline(file=file, user_id=str(user_id))
    return SuccessResponse(data=saved_avatar)


@router.get("/me", response_model=SuccessResponse)
async def get_my_avatar(user_id: UUID = Depends(get_current_user_id)):
    logger.info(f"Get avatar — user={user_id}")

    result = supabase.table("avatars")\
        .select("*")\
        .eq("user_id", str(user_id))\
        .maybe_single()\
        .execute()

    if not result or not result.data:
        raise AvatarNotFoundError()

    return SuccessResponse(data=result.data)

@router.put("/me", response_model=SuccessResponse)
async def update_avatar(
    body: UpdateAvatarRequest,
    user_id: UUID = Depends(get_current_user_id)
):
    logger.info(f"Update avatar — user={user_id}")

    updates = {k: v for k, v in body.model_dump().items() if v is not None}

    if not updates:
        return SuccessResponse(data={"message": "Nothing to update"})

    result = supabase.table("avatars")\
        .update(updates)\
        .eq("user_id", str(user_id))\
        .execute()

    if not result.data:
        raise AvatarNotFoundError()

    return SuccessResponse(data=result.data[0])


@router.delete("/me", response_model=SuccessResponse)
async def delete_avatar(user_id: UUID = Depends(get_current_user_id)):
    logger.info(f"Delete avatar — user={user_id}")

    result = supabase.table("avatars")\
        .select("original_photo_url, processed_photo_url")\
        .eq("user_id", str(user_id))\
        .single()\
        .execute()

    if not result.data:
        raise AvatarNotFoundError()

    avatar = result.data

    supabase.table("avatars")\
        .delete()\
        .eq("user_id", str(user_id))\
        .execute()

    for url_key in ["original_photo_url", "processed_photo_url"]:
        url = avatar.get(url_key)
        if url:
            key = url.split(".amazonaws.com/")[-1]
            deleted = storage.delete(key)
            if not deleted:
                logger.warning(f"Failed to delete S3 image: {key}")

    return SuccessResponse(data={"message": "Avatar deleted successfully"})

@router.get("/asset")
async def get_avatar_asset(user_id: UUID = Depends(get_current_user_id)):
    logger.info(f"Get avatar asset — user={user_id}")
    asset = build_avatar_asset(str(user_id))
    return SuccessResponse(data=asset)