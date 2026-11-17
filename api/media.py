"""
Media API - Try-On v2 T2-1 B5 (2026-11-18).

POST /api/media/sign   {"keys": ["tryons/....png", ...]}   (1-50 keys or stored URLs)
  200 {"items": [{"key", "url", "expires_at"}], "denied": ["..."]}
      expires_at: ISO time the link stops working (null = public, never expires)
  400 KEYS_REQUIRED | TOO_MANY_KEYS
Only the caller's own private images are signed; see pipelines/media_pipeline.py.
"""

from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from shared.dependencies import get_current_user_id
from shared.models.responses import SuccessResponse
from pipelines.media_pipeline import sign_keys

router = APIRouter(prefix="/media", tags=["Media"])


class SignRequest(BaseModel):
    keys: list[str] = Field(default_factory=list, max_length=200)


@router.post("/sign")
def post_sign(body: SignRequest, user_id: UUID = Depends(get_current_user_id)):
    return SuccessResponse(data=sign_keys(user_id, body.keys))