from fastapi import APIRouter, UploadFile, File, Depends
from pydantic import BaseModel
from uuid import UUID
from starlette.concurrency import run_in_threadpool
from shared.logger import get_logger
from shared.models.responses import SuccessResponse
from shared.exceptions import AvatarNotFoundError
from core.storage.supabase_client import supabase
from pipelines.avatar_pipeline import run_avatar_pipeline
from pipelines.avatar_delete_pipeline import delete_avatar_everything
from pipelines.avatar_landmarks import with_landmarks
from shared.dependencies import get_current_user_id


logger = get_logger(__name__)

router = APIRouter(prefix="/avatar", tags=["Avatar"])


class UpdateAvatarRequest(BaseModel):
    """Editable avatar fields.

    Try-On v2 B7 (P8): display_body_shape / display_skin_tone were accepted here
    but those columns don't exist, so sending them gave a 500. They're gone;
    an old client that still sends them is simply ignored (unknown fields are
    dropped), never an error.
    """
    display_height: str | None = None
    hair_style: str | None = None
    hair_color: str | None = None


@router.get("/health")
async def health():
    return {"service": "avatar", "version": "1.3.0", "status": "healthy"}


@router.post("/upload", response_model=SuccessResponse)
async def upload_avatar(
    file: UploadFile = File(...),
    user_id: UUID = Depends(get_current_user_id)
):
    logger.info(f"Avatar upload - user={user_id}")
    saved_avatar = await run_avatar_pipeline(file=file, user_id=str(user_id))
    return SuccessResponse(data=with_landmarks(saved_avatar))


@router.get("/me", response_model=SuccessResponse)
async def get_my_avatar(user_id: UUID = Depends(get_current_user_id)):
    """The owner's avatar. Try-On Studio B10: + `landmarks` (body points on
    processed_photo_url, see pipelines/avatar_landmarks.py). Owner only."""
    logger.info(f"Get avatar - user={user_id}")

    result = supabase.table("avatars")\
        .select("*")\
        .eq("user_id", str(user_id))\
        .maybe_single()\
        .execute()

    if not result or not result.data:
        raise AvatarNotFoundError()

    return SuccessResponse(data=with_landmarks(result.data))


@router.put("/me", response_model=SuccessResponse)
async def update_avatar(
    body: UpdateAvatarRequest,
    user_id: UUID = Depends(get_current_user_id)
):
    logger.info(f"Update avatar - user={user_id}")

    updates = {k: v for k, v in body.model_dump().items() if v is not None}

    if not updates:
        return SuccessResponse(data={"message": "Nothing to update"})

    result = supabase.table("avatars")\
        .update(updates)\
        .eq("user_id", str(user_id))\
        .execute()

    if not result.data:
        raise AvatarNotFoundError()

    return SuccessResponse(data=with_landmarks(result.data[0]))


@router.delete("/me", response_model=SuccessResponse)
async def delete_avatar(user_id: UUID = Depends(get_current_user_id)):
    """
    Privacy delete: avatar photos, body profile, every try-on render and
    every saved look of this user. 404 AVATAR_NOT_FOUND if there is no avatar.
    """
    logger.info(f"Delete avatar - user={user_id}")
    result = await run_in_threadpool(delete_avatar_everything, str(user_id))
    return SuccessResponse(data=result)