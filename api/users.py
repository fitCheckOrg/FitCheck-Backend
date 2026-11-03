"""
Users API (Discover part 2). Thin routes - work lives in pipelines/users_pipeline.py.

B9 handles (2026-10-31), N1 first pick has no lock (2026-11-02):
GET   /api/users/handle-available?handle=   -> {handle, available, reason, suggestion}
GET   /api/users/me                         -> user + handle_can_change_at
PATCH /api/users/me      {"handle"}         -> user + handle_can_change_at
GET   /api/users/by-handle/{handle}         -> user

user = {user_id, display_name, handle, avatar_url, avatar_url_small, is_following, is_me}

B10 public profile (2026-11-02):
GET   /api/users/{user_id}/public
GET   /api/users/by-handle/{handle}/public
GET   /api/users/me/public
PATCH /api/users/me/public  {bio?, city?, style_tags?}
GET   /api/users/{user_id}/posts?cursor=&limit=&include_private=   -> {items, next_cursor}

B10.4 public profile photo (2026-11-03):
POST   /api/users/me/photo   multipart: photo (file), crop_x?, crop_y?, crop_size? (form ints)
DELETE /api/users/me/photo
  -> {avatar_url, avatar_url_small}

Errors: 400 HANDLE_INVALID | 400 HANDLE_RESERVED | 409 HANDLE_TAKEN |
        429 HANDLE_CHANGE_TOO_SOON | 404 USER_NOT_FOUND |
        400 BIO_TOO_LONG | 400 CITY_INVALID | 400 STYLE_TAG_INVALID | 400 INVALID_CURSOR |
        400 PHOTO_INVALID | 413 PHOTO_TOO_LARGE
"""

from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from pydantic import BaseModel, Field

from shared.dependencies import get_current_user_id
from shared.models.responses import SuccessResponse
from pipelines.users_pipeline import (
    check_available, get_me, set_handle, get_by_handle,
    public_profile, public_profile_by_handle, update_public_profile, user_posts,
)
from pipelines.profile_photo_pipeline import upload_profile_photo, delete_profile_photo, MAX_BYTES

router = APIRouter(prefix="/users", tags=["Users"])


class UpdateMeRequest(BaseModel):
    handle: str


class UpdatePublicProfileRequest(BaseModel):
    bio: str | None = None
    city: str | None = None
    style_tags: list[str] | None = Field(default=None, max_length=20)


@router.get("/health")
def health():
    return {"service": "users", "version": "1.2.0", "status": "healthy"}


@router.get("/handle-available")
def handle_available(
    handle: str = Query(..., max_length=40),
    user_id: UUID = Depends(get_current_user_id),
):
    return SuccessResponse(data=check_available(handle, user_id))


@router.get("/me")
def read_me(user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=get_me(user_id))


@router.patch("/me")
def update_me(body: UpdateMeRequest, user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=set_handle(user_id, body.handle))


@router.get("/by-handle/{handle}")
def read_by_handle(handle: str, user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=get_by_handle(handle, user_id))


# ---------------------------------------------------------------- B10.4 photo
# Declared before the /{target_id}/... routes.

@router.post("/me/photo")
def upload_my_photo(
    photo: UploadFile = File(...),
    crop_x: int | None = Form(None),
    crop_y: int | None = Form(None),
    crop_size: int | None = Form(None),
    user_id: UUID = Depends(get_current_user_id),
):
    raw = photo.file.read(MAX_BYTES + 1)          # one byte over -> PHOTO_TOO_LARGE
    crop = {"x": crop_x, "y": crop_y, "size": crop_size}
    return SuccessResponse(data=upload_profile_photo(user_id, raw, crop))


@router.delete("/me/photo")
def delete_my_photo(user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=delete_profile_photo(user_id))


# ---------------------------------------------------------------- B10

@router.get("/me/public")
def read_my_public_profile(user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=public_profile(user_id, user_id))


@router.patch("/me/public")
def edit_my_public_profile(
    body: UpdatePublicProfileRequest,
    user_id: UUID = Depends(get_current_user_id),
):
    sent = {k: getattr(body, k) for k in body.model_fields_set}
    return SuccessResponse(data=update_public_profile(user_id, sent))


@router.get("/by-handle/{handle}/public")
def read_public_profile_by_handle(handle: str, user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=public_profile_by_handle(handle, user_id))


@router.get("/{target_id}/public")
def read_public_profile(target_id: UUID, user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=public_profile(target_id, user_id))


@router.get("/{target_id}/posts")
def read_user_posts(
    target_id: UUID,
    cursor: str | None = Query(None),
    limit: int = Query(20, ge=1, le=50),
    include_private: bool = Query(False),
    user_id: UUID = Depends(get_current_user_id),
):
    return SuccessResponse(data=user_posts(target_id, user_id, cursor, limit, include_private))