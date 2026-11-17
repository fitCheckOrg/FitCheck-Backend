"""
Try-On v2 T2-1 B4 (2026-11-17): push when a background render ends.

Data-only FCM messages (no title / body), sent through the PUSH-1 devices and
Firebase app (pipelines/push_pipeline.py):

    {"type": "render_ready",  "job_id": "...", "outfit_id": "..."}
    {"type": "render_failed", "job_id": "...", "outfit_id": "...", "code": "provider_error"}

    collapse key: render:<job_id>   (one notification per job at most)

Rules (design doc section 7):
    * Sent by the render worker once, when a job SUCCEEDS or FAILS.
      Never for cancelled jobs or instant cache hits (those never reach the worker).
    * No quiet hours and no setting: it's the result of the user's own tap, and silent.
      The app decides what to show (a local notification only when it's in the background).
    * Tokens are read at SEND time for the job's owner, so after a logout / account
      switch the old account's renders never reach this phone.
    * Nothing about the render itself (no image, no URL) is in the push: the app
      reads the job to get it.
    * Best effort: a push failure never changes the job. The result is written to
      render_jobs.notified for support / tests ("render_ready: 1 device", "no_device", ...).
"""

import logging

from core.storage.supabase_client import supabase

logger = logging.getLogger(__name__)


def _message(token: str, data: dict, collapse: str):
    from firebase_admin import messaging
    return messaging.Message(
        token=token,
        data={k: str(v) for k, v in data.items() if v is not None},
        android=messaging.AndroidConfig(collapse_key=collapse[:64], priority="high"),
        apns=messaging.APNSConfig(
            headers={"apns-collapse-id": collapse[:64], "apns-push-type": "background",
                     "apns-priority": "5"},
            payload=messaging.APNSPayload(aps=messaging.Aps(content_available=True)),
        ),
    )


def _deliver(user_id: str, data: dict, collapse: str) -> str:
    """Send to every device of the user. Returns a short result for render_jobs.notified."""
    from firebase_admin import messaging
    from pipelines.push_pipeline import _firebase, _tokens, _drop_token, _is_dead_token
    app = _firebase()
    if app is None:
        return "push_disabled"
    tokens = _tokens(user_id)
    if not tokens:
        return "no_device"
    result = messaging.send_each([_message(t, data, collapse) for t in tokens], app=app)
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
            logger.warning("render push failed user=%s: %s", user_id, resp.exception)
    if delivered:
        return f"{data['type']}: {delivered} device{'' if delivered == 1 else 's'}"
    return f"failed: {last_error or 'send_failed'}"


def notify(job: dict, ok: bool, code: str | None = None) -> None:
    """Called by the worker when a job ends. Never raises."""
    job_id = str(job.get("job_id"))
    try:
        data = {"type": "render_ready" if ok else "render_failed", "job_id": job_id,
                "outfit_id": str(job.get("outfit_id")), "code": None if ok else (code or "provider_error")}
        result = _deliver(str(job["user_id"]), data, f"render:{job_id}")
    except Exception as exc:
        logger.exception("render push crashed job=%s", job_id)
        result = f"failed: {type(exc).__name__}"
    logger.info("render push job=%s -> %s", job_id, result)
    try:
        supabase.table("render_jobs").update({"notified": result[:80]}).eq("job_id", job_id).execute()
    except Exception:
        pass