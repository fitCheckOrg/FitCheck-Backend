"""
Push notifications (PUSH-1, 2026-11-06). Work lives in pipelines/push_pipeline.py.

POST   /api/users/me/devices               {token, platform: android|ios, app_version?, tz?}
         -> {registered: true, platform, tz}   (a token on another account moves to you)
DELETE /api/users/me/devices/{token}       -> {registered: false}   (call on sign-out)
GET    /api/users/me/notification-settings -> settings
PATCH  /api/users/me/notification-settings {follows?, comments?, mentions?, reactions?,
         reactions_mode? instant|digest, quiet_hours?, quiet_start? 0-23, quiet_end? 0-23, tz?}

settings = {follows, comments, mentions, reactions, reactions_mode, quiet_hours,
            quiet_start, quiet_end, tz}

Errors: 400 DEVICE_TOKEN_INVALID | PLATFORM_INVALID | SETTING_INVALID
"""

from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from shared.dependencies import get_current_user_id
from shared.models.responses import SuccessResponse
from pipelines.push_pipeline import (
    register_device, unregister_device, get_settings, update_settings,
)

router = APIRouter(prefix="/users/me", tags=["Push"])


class DeviceRequest(BaseModel):
    token: str
    platform: str
    app_version: str | None = None
    tz: str | None = None


class SettingsRequest(BaseModel):
    follows: bool | None = None
    comments: bool | None = None
    mentions: bool | None = None
    reactions: bool | None = None
    reactions_mode: str | None = None
    quiet_hours: bool | None = None
    quiet_start: int | None = None
    quiet_end: int | None = None
    tz: str | None = None


@router.post("/devices")
def post_device(body: DeviceRequest, user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=register_device(user_id, body.token, body.platform,
                                                body.app_version, body.tz))


@router.delete("/devices/{token}")
def delete_device(token: str, user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=unregister_device(user_id, token))


@router.get("/notification-settings")
def read_settings(user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=get_settings(user_id))


@router.patch("/notification-settings")
def patch_settings(body: SettingsRequest, user_id: UUID = Depends(get_current_user_id)):
    sent = {k: getattr(body, k) for k in body.model_fields_set}
    return SuccessResponse(data=update_settings(user_id, sent))