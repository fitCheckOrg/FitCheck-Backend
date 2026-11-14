"""
ONB-4 C: "Make it yours" setup progress on the server (2026-11-12), so a
killed app, a reinstall or a second phone resumes at the exact step.

Stored on profiles (see the ONB-4 Stage 1 SQL):
    setup_status      in_progress | complete | legacy
    setup_step        current step id
    setup_completed   step ids done
    setup_skipped     step ids skipped
    setup_intents     what the user wants FitCheck for (order kept: the first decides
                      where Ready lands)
    setup_discover    what they'd love to see
    setup_updated_at

Rules:
    * Accounts that existed when the SQL ran are `legacy` and NEVER become
      in_progress (the app shows them the Home "Finish setting up" card).
    * New accounts start `in_progress` at step `handle` (column default).
    * status only ever moves in_progress -> complete. PUT on a complete or legacy
      account may still change the lists (Home edit mode); status stays.
    * A step can't be both completed and skipped: completed wins.
    * Profile fields (handle, photo, bio, city, style words, name) stay in their
      own endpoints; setup only tracks the journey and the two lists.
    * Every write clears the Node service's /auth/me cache (setup_status shows
      up there at once).
"""

import logging
from datetime import datetime, timezone

from core.storage.supabase_client import supabase
from shared.exceptions import FitCheckException
from pipelines.profile_cache import invalidate_profile

logger = logging.getLogger(__name__)

STEPS = ("handle", "photo", "about", "intent", "style", "discover", "people", "know", "notifications")
INTENTS = ("share", "inspiration", "organise", "improve")
DISCOVER = ("outfit_inspiration", "everyday", "creators", "styling_tips",
            "wardrobe_ideas", "photography", "new_collections", "communities")
FIELDS = ("setup_status, setup_step, setup_completed, setup_skipped, "
          "setup_intents, setup_discover, setup_updated_at")


def _bad(message: str) -> FitCheckException:
    return FitCheckException(message, code="SETUP_INVALID", status_code=400)


def _row(uid: str) -> dict:
    rows = supabase.table("profiles").select(FIELDS).eq("id", uid).limit(1).execute().data
    if not rows:
        raise FitCheckException("User not found.", code="USER_NOT_FOUND", status_code=404)
    return rows[0]


def _shape(row: dict) -> dict:
    return {
        "status": row.get("setup_status") or "legacy",
        "step": row.get("setup_step"),
        "completed": row.get("setup_completed") or [],
        "skipped": row.get("setup_skipped") or [],
        "intents": row.get("setup_intents") or [],
        "discover": row.get("setup_discover") or [],
        "updated_at": row.get("setup_updated_at"),
    }


def _ids(values, allowed: tuple, what: str) -> list[str]:
    """Lower-case, known ids only, duplicates dropped, order kept."""
    if values is None:
        return []
    if not isinstance(values, list):
        raise _bad(f"{what} must be a list.")
    out = []
    for v in values:
        v = str(v or "").strip().lower()
        if v not in allowed:
            raise _bad(f"Unknown {what} value: {v or 'empty'}. Allowed: {', '.join(allowed)}.")
        if v not in out:
            out.append(v)
    return out


def get_setup(user_id) -> dict:
    return _shape(_row(str(user_id)))


def update_setup(user_id, sent: dict) -> dict:
    """PUT: any subset of step, completed, skipped, intents, discover.
    A list that is sent replaces the stored list."""
    uid = str(user_id)
    row = _row(uid)
    updates = {}

    if sent.get("step") is not None:
        step = str(sent["step"]).strip().lower()
        if step not in STEPS:
            raise _bad(f"Unknown step: {step}. Allowed: {', '.join(STEPS)}.")
        updates["setup_step"] = step

    completed = _ids(sent["completed"], STEPS, "completed") if "completed" in sent \
        else (row.get("setup_completed") or [])
    skipped = _ids(sent["skipped"], STEPS, "skipped") if "skipped" in sent \
        else (row.get("setup_skipped") or [])
    if "completed" in sent or "skipped" in sent:
        updates["setup_completed"] = completed
        updates["setup_skipped"] = [s for s in skipped if s not in completed]   # completed wins

    if "intents" in sent:
        updates["setup_intents"] = _ids(sent["intents"], INTENTS, "intents")
    if "discover" in sent:
        updates["setup_discover"] = _ids(sent["discover"], DISCOVER, "discover")

    if updates:
        updates["setup_updated_at"] = datetime.now(timezone.utc).isoformat()
        supabase.table("profiles").update(updates).eq("id", uid).execute()
        invalidate_profile(uid)
    return get_setup(uid)


def complete_setup(user_id) -> dict:
    """in_progress -> complete. legacy stays legacy, complete stays complete."""
    uid = str(user_id)
    row = _row(uid)
    if row.get("setup_status") == "in_progress":
        supabase.table("profiles").update({
            "setup_status": "complete",
            "setup_updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", uid).eq("setup_status", "in_progress").execute()
        invalidate_profile(uid)
        logger.info("setup complete user=%s", uid)
    return get_setup(uid)