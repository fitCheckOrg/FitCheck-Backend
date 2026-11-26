"""
Developer-only routes (Try-On Studio B13, 2026-11-26).

Only accounts listed in DEV_USER_IDS (.env, comma-separated user ids) can use these;
everyone else gets 404, as if the routes didn't exist.

GET  /api/dev/fashn-credits          -> credit status (balance, %, level, last check, alerts)
POST /api/dev/fashn-credits/check    -> check FASHN's balance now, then the status
POST /api/dev/renders/pause {"paused": true|false}
                                     -> stop / resume new paid renders by hand
"""

import os
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from shared.dependencies import get_current_user_id
from shared.exceptions import FitCheckException
from shared.models.responses import SuccessResponse
from pipelines.fashn_credits import status, run_check, set_manual_pause

router = APIRouter(prefix="/dev", tags=["Developer"])


class PauseRequest(BaseModel):
    paused: bool


def _require_dev(user_id: UUID) -> None:
    allowed = {s.strip().lower() for s in (os.getenv("DEV_USER_IDS") or "").split(",") if s.strip()}
    if str(user_id).lower() not in allowed:
        raise FitCheckException("Not found.", code="NOT_FOUND", status_code=404)


@router.get("/fashn-credits")
def read_credits(user_id: UUID = Depends(get_current_user_id)):
    _require_dev(user_id)
    return SuccessResponse(data=status())


@router.post("/fashn-credits/check")
def check_credits(user_id: UUID = Depends(get_current_user_id)):
    _require_dev(user_id)
    return SuccessResponse(data=run_check("manual"))


@router.post("/renders/pause")
def pause_renders(body: PauseRequest, user_id: UUID = Depends(get_current_user_id)):
    _require_dev(user_id)
    return SuccessResponse(data=set_manual_pause(body.paused))