"""
PUSH-1 Push notifications (2026-11-06). Firebase Cloud Messaging (HTTP v1 via
firebase-admin). Mirrors the bell (D15): a push is only ever sent for an
activity row the receiver can see.

Tables (see the PUSH-1 SQL):
    user_devices          (token pk, user_id, platform, app_version, tz, created_at, last_seen_at)
    notification_settings (user_id pk, follows, comments, mentions, reactions,
                           reactions_mode instant|digest, quiet_hours, quiet_start,
                           quiet_end, tz, updated_at)
    push_queue            (push_id, user_id, activity_id, type, send_after,
                           status pending|sending|sent|skipped|failed, reason, sent_at)

Approved rules (Keji, 2026-11-06):
    * Credentials: service-account JSON on the server only, path in
      FIREBASE_SERVICE_ACCOUNT_FILE. Not set -> push is off, nothing is queued.
    * A device token belongs to ONE user. Registering it again moves it to the
      new user; sign-out deletes it. Pushes go to the receiver's tokens at SEND
      time, so after a logout / account switch the old account gets nothing.
    * Reactions (reacted, liked_comment) are a digest: at most one push per
      activity row (= per post per day) per hour, built from the latest
      counts ("Ada and 4 others reacted to your look").
    * Quiet hours (default 22:00-07:00, user's timezone): EVERYTHING due inside
      the window is queued until quiet_end; several due at once for one user
      become one summary push.
    * Follows, comments, mentions: immediate outside quiet hours.
    * Planner reminders are local notifications on the phone - not here.
    * No images, no avatar, no person photo; text is only names + the event.
      People the receiver blocked / hid (or who blocked them) never appear.
    * Tokens FCM reports as unregistered / invalid are deleted.
"""

import asyncio
import logging
import os
import threading
from datetime import datetime, timedelta, timezone

from postgrest.exceptions import APIError

from core.storage.supabase_client import supabase

logger = logging.getLogger(__name__)

try:                                     # pick up .env if python-dotenv is installed
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

PLATFORMS = ("android", "ios")
REACTION_MODES = ("instant", "digest")
DIGEST_TYPES = {"reacted", "liked_comment"}
DIGEST_WINDOW = timedelta(hours=1)
SETTING_FOR_TYPE = {
    "followed": "follows",
    "commented": "comments", "replied": "comments",
    "mentioned": "mentions",
    "reacted": "reactions", "liked_comment": "reactions", "milestone": "reactions",
}
DEFAULT_SETTINGS = {
    "follows": True, "comments": True, "mentions": True, "reactions": True,
    "reactions_mode": "digest", "quiet_hours": True, "quiet_start": 22, "quiet_end": 7,
    "tz": None,
}
DEFAULT_TZ = os.getenv("PUSH_DEFAULT_TZ") or "Africa/Lagos"
TICK_SECONDS = 15
BATCH = 200
MAX_TOKEN_LEN = 4096

_app = None
_disabled_logged = False


# ---------------------------------------------------------------- helpers

def _bad(code: str, message: str, status: int = 400):
    from shared.exceptions import FitCheckException
    return FitCheckException(message, code=code, status_code=status)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse(ts) -> datetime:
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))


def _zone(name: str | None):
    from zoneinfo import ZoneInfo
    for candidate in (name, DEFAULT_TZ, "UTC"):
        if not candidate:
            continue
        try:
            return ZoneInfo(candidate)
        except Exception:
            continue
    return timezone.utc


def valid_tz(name) -> str | None:
    """The IANA name if Python knows it (needs the tzdata package on Windows)."""
    if not name:
        return None
    from zoneinfo import ZoneInfo
    try:
        ZoneInfo(str(name))
        return str(name)
    except Exception:
        return None


def enabled() -> bool:
    return bool(os.getenv("FIREBASE_SERVICE_ACCOUNT_FILE"))


def _firebase():
    """Initialise firebase-admin once. None when push is not configured."""
    global _app, _disabled_logged
    if _app is not None:
        return _app
    path = os.getenv("FIREBASE_SERVICE_ACCOUNT_FILE")
    if not path:
        if not _disabled_logged:
            logger.warning("push disabled: FIREBASE_SERVICE_ACCOUNT_FILE not set")
            _disabled_logged = True
        return None
    import firebase_admin
    from firebase_admin import credentials
    _app = firebase_admin.initialize_app(credentials.Certificate(path), name="fitcheck-push")
    logger.info("push enabled (firebase project %s)", _app.project_id)
    return _app


# ---------------------------------------------------------------- devices

def register_device(user_id, token, platform, app_version=None, tz=None) -> dict:
    me = str(user_id)
    token = str(token or "").strip()
    if not token or len(token) > MAX_TOKEN_LEN:
        raise _bad("DEVICE_TOKEN_INVALID", "That device token isn't valid.")
    platform = str(platform or "").strip().lower()
    if platform not in PLATFORMS:
        raise _bad("PLATFORM_INVALID", "platform must be 'android' or 'ios'.")
    tz_name = valid_tz(tz)
    now = _iso(_now())
    # Upsert by token: a token on another account moves to this one.
    supabase.table("user_devices").upsert({
        "token": token, "user_id": me, "platform": platform,
        "app_version": (str(app_version)[:40] if app_version else None),
        "tz": tz_name, "last_seen_at": now,
    }, on_conflict="token").execute()
    if tz_name:
        _save_settings(me, {"tz": tz_name})
    logger.info("push device registered user=%s platform=%s", me, platform)
    return {"registered": True, "platform": platform, "tz": tz_name}


def unregister_device(user_id, token) -> dict:
    supabase.table("user_devices").delete()\
        .eq("token", str(token or "").strip()).eq("user_id", str(user_id)).execute()
    return {"registered": False}


def _tokens(user_id: str) -> list[str]:
    rows = supabase.table("user_devices").select("token").eq("user_id", user_id).execute().data or []
    return [r["token"] for r in rows]


def _drop_token(token: str) -> None:
    supabase.table("user_devices").delete().eq("token", token).execute()


# ---------------------------------------------------------------- settings

def get_settings(user_id) -> dict:
    rows = supabase.table("notification_settings").select("*")\
        .eq("user_id", str(user_id)).limit(1).execute().data
    row = rows[0] if rows else {}
    out = {k: (row.get(k) if row.get(k) is not None else v) for k, v in DEFAULT_SETTINGS.items()}
    if not out["tz"]:
        out["tz"] = DEFAULT_TZ
    return out


def _save_settings(user_id: str, updates: dict) -> None:
    supabase.table("notification_settings").upsert(
        {"user_id": user_id, **updates, "updated_at": _iso(_now())}, on_conflict="user_id"
    ).execute()


def update_settings(user_id, sent: dict) -> dict:
    me = str(user_id)
    updates = {}
    for key in ("follows", "comments", "mentions", "reactions", "quiet_hours"):
        if key in sent and sent[key] is not None:
            updates[key] = bool(sent[key])
    if sent.get("reactions_mode") is not None:
        mode = str(sent["reactions_mode"]).lower()
        if mode not in REACTION_MODES:
            raise _bad("SETTING_INVALID", "reactions_mode must be 'instant' or 'digest'.")
        updates["reactions_mode"] = mode
    for key in ("quiet_start", "quiet_end"):
        if sent.get(key) is not None:
            hour = int(sent[key])
            if not 0 <= hour <= 23:
                raise _bad("SETTING_INVALID", f"{key} must be an hour from 0 to 23.")
            updates[key] = hour
    if sent.get("tz") is not None:
        tz_name = valid_tz(sent["tz"])
        if not tz_name:
            raise _bad("SETTING_INVALID", "tz must be a timezone name like Africa/Lagos.")
        updates["tz"] = tz_name
    if updates:
        _save_settings(me, updates)
    return get_settings(me)


# ---------------------------------------------------------------- quiet hours

def _quiet_until(settings: dict, at: datetime) -> datetime | None:
    """If `at` falls in the user's quiet hours, when they end (UTC). Else None."""
    if not settings.get("quiet_hours"):
        return None
    start, end = int(settings["quiet_start"]), int(settings["quiet_end"])
    if start == end:
        return None
    local = at.astimezone(_zone(settings.get("tz")))
    hour = local.hour
    inside = (hour >= start or hour < end) if start > end else (start <= hour < end)
    if not inside:
        return None
    end_local = local.replace(hour=end, minute=0, second=0, microsecond=0)
    if end_local <= local:
        end_local += timedelta(days=1)
    return end_local.astimezone(timezone.utc)


# ---------------------------------------------------------------- enqueue (called by activity)

def enqueue(recipient, activity_id, type_: str) -> None:
    """Called after the bell row was created / got a new actor. Runs in a
    background thread so the action (follow, react, comment) never waits."""
    if not enabled() or not activity_id:
        return
    threading.Thread(target=_enqueue, args=(recipient, activity_id, type_), daemon=True).start()


def _enqueue(recipient, activity_id, type_: str) -> None:
    """Best effort: never raises."""
    try:
        uid, aid = str(recipient), str(activity_id)
        settings = get_settings(uid)
        if not settings.get(SETTING_FOR_TYPE.get(type_, ""), False):
            return
        pending = supabase.table("push_queue").select("push_id")\
            .eq("activity_id", aid).in_("status", ["pending", "sending"]).limit(1).execute().data
        if pending:
            return                       # will go out with the latest counts
        send_after = _now()
        if type_ in DIGEST_TYPES and settings.get("reactions_mode") == "digest":
            last = supabase.table("push_queue").select("sent_at")\
                .eq("activity_id", aid).eq("status", "sent")\
                .order("sent_at", desc=True).limit(1).execute().data
            if last and last[0].get("sent_at"):
                nxt = _parse(last[0]["sent_at"]) + DIGEST_WINDOW
                send_after = max(send_after, nxt)
        quiet_end = _quiet_until(settings, send_after)
        if quiet_end:
            send_after = quiet_end
        try:
            supabase.table("push_queue").insert({
                "user_id": uid, "activity_id": aid, "type": type_,
                "send_after": _iso(send_after),
            }).execute()
        except APIError as exc:
            if getattr(exc, "code", None) != "23505":       # one pending per activity
                raise
    except Exception:
        logger.exception("push enqueue failed type=%s to=%s", type_, recipient)


# ---------------------------------------------------------------- building messages

def _name(user: dict | None) -> str:
    if not user:
        return "Someone"
    return user.get("display_name") or (f"@{user['handle']}" if user.get("handle") else "Someone")


def _who(names: list[str], count: int) -> str:
    if not names:
        return "Someone"
    others = max(0, count - 1)
    if others == 0:
        return names[0]
    return f"{names[0]} and {others} other{'' if others == 1 else 's'}"


def _deliverable(rows: list[dict], viewer: str) -> list[dict]:
    """Activity rows -> [{activity, type, actors(names), count, post_id, actor_id}]
    only for rows the receiver could see in the bell right now."""
    from pipelines.safety_pipeline import excluded_user_ids
    from pipelines.users_pipeline import build_users

    excluded = excluded_user_ids(viewer)
    post_ids = list({str(r["post_id"]) for r in rows if r.get("post_id")})
    comment_ids = list({str(r["comment_id"]) for r in rows if r.get("comment_id")})
    posts, comments = {}, {}
    if post_ids:
        for p in supabase.table("outfits").select("outfit_id, user_id, is_public")\
                .in_("outfit_id", post_ids).execute().data or []:
            posts[str(p["outfit_id"])] = p
    if comment_ids:
        for c in supabase.table("comments").select("comment_id, deleted")\
                .in_("comment_id", comment_ids).execute().data or []:
            comments[str(c["comment_id"])] = c

    visible = {r["activity_id"]: [str(a) for a in (r.get("actor_ids") or []) if str(a) not in excluded]
               for r in rows}
    people = build_users([ids[0] for ids in visible.values() if ids], viewer)

    out = []
    for r in rows:
        actors = visible[r["activity_id"]]
        if r["type"] != "milestone" and not actors:
            continue
        if r.get("post_id"):
            p = posts.get(str(r["post_id"]))
            if not p or (not p.get("is_public") and str(p.get("user_id")) != viewer):
                continue
        if r.get("comment_id"):
            c = comments.get(str(r["comment_id"]))
            if not c or c.get("deleted"):
                continue
        hidden = len([a for a in (r.get("actor_ids") or []) if str(a) in excluded])
        out.append({
            "activity": r, "type": r["type"],
            "names": [_name(people.get(actors[0]))] if actors else [],
            "count": max(len(actors), int(r.get("actor_count") or 0) - hidden),
            "post_id": str(r["post_id"]) if r.get("post_id") else None,
            "actor_id": actors[0] if actors else None,
        })
    return out


def _text(item: dict) -> tuple[str, str]:
    t, who = item["type"], _who(item["names"], item["count"])
    if t == "followed":
        return "New follower", f"{who} started following you"
    if t == "reacted":
        return "New reactions", f"{who} reacted to your look"
    if t == "commented":
        return "New comment", f"{who} commented on your look"
    if t == "replied":
        return "New reply", f"{who} replied to your comment"
    if t == "mentioned":
        where = "a comment" if item["activity"].get("comment_id") else "a post"
        return "You were mentioned", f"{who} mentioned you in {where}"
    if t == "liked_comment":
        return "Comment liked", f"{who} liked your comment"
    if t == "milestone":
        value = (item["activity"].get("milestone") or {}).get("value")
        return "Milestone", f"Your look reached {value} reactions" if value else "Your look hit a milestone"
    return "FitCheck", "You have new activity"


def _summary(items: list[dict]) -> tuple[str, str]:
    followers = sum(i["count"] for i in items if i["type"] == "followed")
    reactions = sum(i["count"] for i in items if i["type"] in ("reacted", "liked_comment"))
    comments = len([i for i in items if i["type"] in ("commented", "replied")])
    mentions = len([i for i in items if i["type"] == "mentioned"])
    parts = []
    for n, word in ((followers, "new follower"), (reactions, "reaction"),
                    (comments, "comment"), (mentions, "mention")):
        if n:
            parts.append(f"{n} {word}{'' if n == 1 else 's'}")
    milestones = len([i for i in items if i["type"] == "milestone"])
    if milestones:
        parts.append(f"{milestones} milestone{'' if milestones == 1 else 's'}")
    return "While you were away", " · ".join(parts) or "You have new activity"


def _message(token: str, title: str, body: str, data: dict, collapse: str):
    from firebase_admin import messaging
    data = {k: str(v) for k, v in data.items() if v is not None}
    return messaging.Message(
        token=token,
        notification=messaging.Notification(title=title, body=body),
        data=data,
        android=messaging.AndroidConfig(
            collapse_key=collapse[:64], priority="high",
            notification=messaging.AndroidNotification(tag=collapse[:64]),
        ),
        apns=messaging.APNSConfig(headers={"apns-collapse-id": collapse[:64]}),
    )


def _is_dead_token(exc) -> bool:
    from firebase_admin import messaging, exceptions
    if isinstance(exc, (messaging.UnregisteredError, messaging.SenderIdMismatchError)):
        return True
    return isinstance(exc, exceptions.InvalidArgumentError) and "registration token" in str(exc).lower()


def _send(user_id: str, title: str, body: str, data: dict, collapse: str) -> tuple[int, str | None]:
    """Send to every token of the user. Returns (delivered count, reason if none)."""
    from firebase_admin import messaging
    app = _firebase()
    if app is None:
        return 0, "push_disabled"
    tokens = _tokens(user_id)
    if not tokens:
        return 0, "no_device"
    msgs = [_message(t, title, body, data, collapse) for t in tokens]
    result = messaging.send_each(msgs, app=app)
    delivered, last_error = 0, None
    for token, resp in zip(tokens, result.responses):
        if resp.success:
            delivered += 1
            continue
        last_error = type(resp.exception).__name__
        if _is_dead_token(resp.exception):
            _drop_token(token)
            logger.info("push token removed user=%s (%s)", user_id, last_error)
        else:
            logger.warning("push send failed user=%s: %s", user_id, resp.exception)
    return delivered, (None if delivered else (last_error or "send_failed"))


# ---------------------------------------------------------------- worker

def _finish(push_ids: list[str], status: str, reason: str | None = None) -> None:
    if push_ids:
        update = {"status": status, "reason": reason}
        if status == "sent":
            update["sent_at"] = _iso(_now())
        supabase.table("push_queue").update(update).in_("push_id", push_ids).execute()


def process_due() -> int:
    """One pass: send everything due. Returns how many queue rows were handled."""
    if _firebase() is None:
        return 0
    due = supabase.table("push_queue").select("push_id, user_id, activity_id, type")\
        .eq("status", "pending").lte("send_after", _iso(_now()))\
        .order("send_after").limit(BATCH).execute().data or []
    if not due:
        return 0
    by_user: dict[str, list[dict]] = {}
    for row in due:
        by_user.setdefault(str(row["user_id"]), []).append(row)

    handled = 0
    for uid, rows in by_user.items():
        claimed = supabase.table("push_queue").update({"status": "sending"})\
            .in_("push_id", [r["push_id"] for r in rows]).eq("status", "pending").execute().data or []
        if not claimed:
            continue
        ids = [r["push_id"] for r in claimed]
        handled += len(ids)
        try:
            settings = get_settings(uid)
            allowed = [r for r in claimed if settings.get(SETTING_FOR_TYPE.get(r["type"], ""), False)]
            activity = supabase.table("activity").select("*")\
                .in_("activity_id", [r["activity_id"] for r in allowed]).execute().data if allowed else []
            items = _deliverable(activity or [], uid)
            if not items:
                _finish(ids, "skipped", "nothing_visible")
                continue
            if len(items) == 1:
                item = items[0]
                title, body = _text(item)
                data = {"type": item["type"], "activity_id": item["activity"]["activity_id"],
                        "post_id": item["post_id"],
                        "user_id": item["actor_id"] if item["type"] == "followed" else None}
                collapse = str(item["activity"]["activity_id"])
            else:
                title, body = _summary(items)
                data = {"type": "summary", "count": len(items)}
                collapse = f"summary-{uid}"
            delivered, reason = _send(uid, title, body, data, collapse)
            _finish(ids, "sent" if delivered else ("skipped" if reason in ("no_device", "push_disabled") else "failed"),
                    None if delivered else reason)
        except Exception:
            logger.exception("push send pass failed user=%s", uid)
            _finish(ids, "failed", "error")
    return handled


async def push_worker() -> None:
    """Background loop started with the app. Off when push isn't configured."""
    if not enabled():
        _firebase()                      # logs "push disabled" once
        return
    logger.info("push worker started (every %ss)", TICK_SECONDS)
    while True:
        try:
            await asyncio.to_thread(process_due)
        except Exception:
            logger.exception("push worker pass failed")
        await asyncio.sleep(TICK_SECONDS)