"""
pipelines/stylist_feedback_pipeline.py

Records what the user does with a Stylist suggestion, so ranking can learn
later. PERS-1 step 1: STORE ONLY - generation does not read these yet.

action:
  saved     user saved the outfit
  skipped   user dismissed / swiped past it
  worn      user marked it as worn
  tried_on  user opened a try-on render of it
  swapped   user replaced one piece (send replaced_item_id + replacement_item_id;
            item_ids = the outfit AFTER the swap)
"""

import logging
from uuid import UUID

from core.storage.supabase_client import supabase
from shared.exceptions import FitCheckException, ClothingItemNotFoundError
from workers.stylist.outfit_key import outfit_key

logger = logging.getLogger(__name__)

TABLE = "stylist_feedback"
ACTIONS = {"saved", "skipped", "worn", "tried_on", "swapped"}
MAX_ITEMS = 10
MAX_TEXT = 40


def _bad(message: str, code: str) -> FitCheckException:
    return FitCheckException(message, code=code, status_code=400)


def _short_text(value: str | None, field: str) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if len(value) > MAX_TEXT:
        raise _bad(f"{field} must be {MAX_TEXT} characters or fewer.", "INVALID_FEEDBACK")
    return value


def record_feedback(
    user_id: UUID,
    action: str,
    item_ids: list[UUID],
    occasion: str | None = None,
    mood: str | None = None,
    source: str | None = None,
    replaced_item_id: UUID | None = None,
    replacement_item_id: UUID | None = None,
) -> dict:
    uid = str(user_id)
    action = (action or "").strip().lower()

    if action not in ACTIONS:
        raise _bad(
            f"action must be one of: {', '.join(sorted(ACTIONS))}.",
            "INVALID_FEEDBACK_ACTION",
        )

    ids = list(dict.fromkeys(str(i) for i in item_ids))  # dedupe, keep order
    if not ids:
        raise _bad("item_ids needs at least 1 item.", "INVALID_FEEDBACK")
    if len(ids) > MAX_ITEMS:
        raise _bad(f"item_ids can have at most {MAX_ITEMS} items.", "INVALID_FEEDBACK")

    replaced = str(replaced_item_id) if replaced_item_id else None
    replacement = str(replacement_item_id) if replacement_item_id else None
    if action == "swapped":
        if not replaced or not replacement:
            raise _bad(
                "A swapped event needs replaced_item_id and replacement_item_id.",
                "INVALID_FEEDBACK",
            )
        if replacement not in ids:
            raise _bad(
                "replacement_item_id must be in item_ids (the outfit after the swap).",
                "INVALID_FEEDBACK",
            )
    elif replaced or replacement:
        raise _bad(
            "replaced_item_id / replacement_item_id are only allowed for action=swapped.",
            "INVALID_FEEDBACK",
        )

    # Every referenced item must belong to this user (archived items are fine:
    # an outfit can be worn and then archived later).
    check_ids = set(ids) | ({replaced} if replaced else set())
    owned = supabase.table("closet_items").select("item_id")\
        .eq("user_id", uid).in_("item_id", list(check_ids)).execute()
    if len({str(r["item_id"]) for r in (owned.data or [])}) != len(check_ids):
        raise ClothingItemNotFoundError()

    row = {
        "user_id": uid,
        "action": action,
        "item_ids": ids,
        "outfit_key": outfit_key(ids),
        "occasion": _short_text(occasion, "occasion"),
        "mood": _short_text(mood, "mood"),
        "source": _short_text(source, "source"),
        "replaced_item_id": replaced,
        "replacement_item_id": replacement,
    }

    res = supabase.table(TABLE).insert(row).execute()
    if not res.data:
        raise FitCheckException("Failed to record feedback.", code="DATABASE_ERROR", status_code=500)

    saved = res.data[0]
    logger.info("stylist feedback user=%s action=%s items=%d", uid, action, len(ids))
    return {
        "event_id": saved["event_id"],
        "action": saved["action"],
        "outfit_key": saved["outfit_key"],
        "created_at": saved["created_at"],
    }