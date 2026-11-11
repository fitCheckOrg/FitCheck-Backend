"""
Push notifications (PUSH-1, 2026-11-06). Work lives in pipelines/push_pipeline.py.

POST   /api/users/me/devices               {token, platform: android|ios, app_version?, tz?}
         -> {registered: true, platform, tz}   (a token on another account moves to you)
DELETE /api/users/me/devices/{token}       -> {registered: false}   (call on sign-out)
GET    /api/users/me/notification-settings -> settings
PATCH  /api/users/me/notification-settings {notes?, follows?, comments?, mentions?, reactions?,
         reactions_mode? instant|digest, quiet_hours?, quiet_start? 0-23, quiet_end? 0-23, tz?}

settings = {follows, comments, mentions, reactions, reactions_mode, quiet_hours,
            quiet_start, quiet_end, tz, notes}

NOTE-PUSH-1 (2026-11-11):
PUT    /api/users/{id}/note-alerts         -> {user_id, note_alerts: true}    (default)
DELETE /api/users/{id}/note-alerts         -> {user_id, note_alerts: false}   (muted, silent)

Errors: 400 DEVICE_TOKEN_INVALID | PLATFORM_INVALID | SETTING_INVALID | CANNOT_MUTE_SELF
        404 USER_NOT_FOUND
"""

from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from shared.dependencies import get_current_user_id
from shared.models.responses import SuccessResponse
from pipelines.push_pipeline import (
    register_device, unregister_device, get_settings, update_settings, set_note_alerts,
)

router = APIRouter(prefix="/users", tags=["Push"])


class DeviceRequest(BaseModel):
    token: str
    platform: str
    app_version: str | None = None
    tz: str | None = None


class SettingsRequest(BaseModel):
    notes: bool | None = None                    # NOTE-PUSH-1
    follows: bool | None = None
    comments: bool | None = None
    mentions: bool | None = None
    reactions: bool | None = None
    reactions_mode: str | None = None
    quiet_hours: bool | None = None
    quiet_start: int | None = None
    quiet_end: int | None = None
    tz: str | None = None


@router.post("/me/devices")
def post_device(body: DeviceRequest, user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=register_device(user_id, body.token, body.platform,
                                                body.app_version, body.tz))


@router.delete("/me/devices/{token}")
def delete_device(token: str, user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=unregister_device(user_id, token))


@router.get("/me/notification-settings")
def read_settings(user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=get_settings(user_id))


@router.patch("/me/notification-settings")
def patch_settings(body: SettingsRequest, user_id: UUID = Depends(get_current_user_id)):
    sent = {k: getattr(body, k) for k in body.model_fields_set}
    return SuccessResponse(data=update_settings(user_id, sent))


# ---------------------------------------------------------------- NOTE-PUSH-1

@router.put("/{target_id}/note-alerts")
def note_alerts_on(target_id: UUID, user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=set_note_alerts(user_id, target_id, True))


@router.delete("/{target_id}/note-alerts")
def note_alerts_off(target_id: UUID, user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=set_note_alerts(user_id, target_id, False))