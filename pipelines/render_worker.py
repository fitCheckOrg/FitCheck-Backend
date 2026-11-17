"""
Try-On v2 T2-1 B2 (2026-11-17): the background render worker.

Started with the app (main.py). Every TICK seconds it asks the database for jobs
(claim_render_jobs: queued ones, plus running ones whose worker died) - never more
than RENDER_CONCURRENCY running at once, across all servers - and runs each one
in a thread:

    for each step (top, then bottom onto the top's result):
        finished in the cache?          -> next step (free)
        someone already submitted it?   -> keep checking THAT request (never re-submit:
                                           a crash never pays twice)
        otherwise                       -> claim the cache row, submit to the provider,
                                           check every POLL_EVERY s, give up after
                                           STEP_TIMEOUT s (provider cancel is attempted)
        done -> download (retries), upload to S3, mark the cache row completed
    all steps done -> job 'succeeded' with the final image

Cancelled while the provider is working: that step is still finished and cached
(it's paid for), the job stays 'cancelled' and nothing more is started.
A job's lease is renewed every pass; if the server dies the next claim picks it up
again (at most MAX_ATTEMPTS times, then 'failed').

B3 (2026-11-17): every generation records the job that sent it (submitted_by_job).
A job is charged only when it SUCCEEDS - finish_render_job() marks it succeeded and
charges each generation it sent, once (render_charges). Failed / cancelled: nothing.

B3 fix (2026-11-17): a network blip no longer fails the job. When the connection
drops (database, S3 or fal) and NOTHING was sent to the provider for the step, the
job is parked for RETRY_DELAY seconds and picked up again (at most MAX_ATTEMPTS
tries in all). Re-running is safe: a step that was already submitted is followed by
its provider_request_id, never submitted again. An error AFTER fal may have
received the request (not a connect error) still fails the step - retrying it
could pay for the same render twice.
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta, timezone

import httpx
import requests
from postgrest.exceptions import APIError

from core.storage.db_retry import DB_CONNECTION_ERRORS
from core.storage.s3 import storage, S3Folders
from core.storage.supabase_client import supabase

logger = logging.getLogger(__name__)

TICK = 2
POLL_EVERY = 3
STEP_TIMEOUT = 120
WAIT_FOR_OTHERS = STEP_TIMEOUT + 60      # waiting for a render someone else is doing
LEASE_SECONDS = 180
MAX_ATTEMPTS = 4                         # claims of one job (network retries + crash recoveries)
RETRY_DELAY = 30                         # seconds a job waits after a network blip
RENDERS = "tryon_renders"
JOBS = "render_jobs"
_running: set[str] = set()


def _concurrency() -> int:
    try:
        return max(1, int(os.getenv("RENDER_CONCURRENCY") or 3))
    except ValueError:
        return 3


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None = None) -> str:
    return (dt or _now()).isoformat()


def _parse(ts) -> datetime | None:
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00")) if ts else None


class _Stop(Exception):
    """The job ended (failed / cancelled); stop working on it."""


class _Retry(Exception):
    """A network blip before anything was paid for; try the job again shortly."""


# Connection problems (database, S3, fal) that are worth waiting out.
_NETWORK_ERRORS = tuple({httpx.TransportError, requests.ConnectionError, requests.Timeout,
                         *DB_CONNECTION_ERRORS})
# Errors that prove the request never reached fal (safe to submit again).
_NOT_CONNECTED = (httpx.ConnectError, httpx.ConnectTimeout, requests.ConnectionError)


def _chain(exc: BaseException):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def _is_network(exc: BaseException) -> bool:
    return any(isinstance(e, _NETWORK_ERRORS) for e in _chain(exc))


def _not_sent(exc: BaseException) -> bool:
    """True when a failed submit certainly never reached the provider: the photos
    couldn't be downloaded (that happens before sending), or fal couldn't be reached."""
    if "could not download" in str(exc):
        return True
    return any(isinstance(e, _NOT_CONNECTED) for e in _chain(exc))


def _park(job_id: str, why: str) -> None:
    """Leave the job running but let the next claim pick it up in RETRY_DELAY s."""
    logger.warning("render job %s will retry in %ss: %s", job_id, RETRY_DELAY, why[:200])
    try:
        supabase.table(JOBS).update({"lease_until": _iso(_now() + timedelta(seconds=RETRY_DELAY))})\
            .eq("job_id", job_id).eq("status", "running").execute()
    except Exception:
        pass        # database still down: the normal lease runs out and the job is picked up anyway


# ---------------------------------------------------------------- job row helpers

def _job(job_id: str) -> dict | None:
    rows = supabase.table(JOBS).select("*").eq("job_id", job_id).limit(1).execute().data
    return rows[0] if rows else None


def _renew(job_id: str) -> str | None:
    """Extend the lease. Returns the job's current status."""
    rows = supabase.table(JOBS).update({"lease_until": _iso(_now() + timedelta(seconds=LEASE_SECONDS))})\
        .eq("job_id", job_id).in_("status", ["running", "cancelled"]).execute().data
    return rows[0]["status"] if rows else None


def _fail(job_id: str, code: str, message: str) -> None:
    logger.warning("render job %s failed: %s %s", job_id, code, message)
    supabase.table(JOBS).update({
        "status": "failed", "failure_code": code, "failure_message": message[:300],
        "finished_at": _iso(), "lease_until": None,
    }).eq("job_id", job_id).eq("status", "running").execute()
    raise _Stop()


# ---------------------------------------------------------------- cache row helpers

def _find(key: dict) -> dict | None:
    q = supabase.table(RENDERS).select("id, status, s3_key, updated_at, provider_request_id, provider_submitted_at")
    for col, val in key.items():
        q = q.eq(col, val)
    rows = q.limit(1).execute().data
    return rows[0] if rows else None


def _claim(row: dict | None, key: dict, uid: str) -> str | None:
    """Become the one who renders this cache row. Row id, or None if someone else is."""
    fresh = {"status": "processing", "error_message": None, "updated_at": _iso(),
             "provider_request_id": None, "provider_submitted_at": None}
    if row is None:
        try:
            return supabase.table(RENDERS).insert({**key, "user_id": uid, **fresh}).execute().data[0]["id"]
        except APIError as exc:
            if getattr(exc, "code", None) == "23505":
                return None
            raise
    if row["status"] == "failed":
        res = supabase.table(RENDERS).update(fresh).eq("id", row["id"]).eq("status", "failed").execute()
    else:                                                       # stale processing row
        cutoff = _iso(_now() - timedelta(seconds=WAIT_FOR_OTHERS))
        res = supabase.table(RENDERS).update(fresh).eq("id", row["id"])\
            .in_("status", ["processing", "pending"]).lt("updated_at", cutoff).execute()
    return row["id"] if res.data else None


def _row_failed(row_id: str, message: str) -> None:
    supabase.table(RENDERS).update({"status": "failed", "error_message": message[:500],
                                    "updated_at": _iso()}).eq("id", row_id).execute()


def _complete(row_id: str, png: bytes, garment_ids: list[str]) -> dict:
    up = storage.upload(png, folder=S3Folders.TRYON, extension="png", content_type="image/png")
    now = _iso()
    supabase.table(RENDERS).update({
        "status": "completed", "s3_key": up["key"], "error_message": None,
        "updated_at": now, "completed_at": now, "garment_ids": garment_ids,
    }).eq("id", row_id).execute()
    return {"id": row_id, "status": "completed", "s3_key": up["key"]}


# ---------------------------------------------------------------- one step

def _step(job: dict, key: dict, person_url: str, garment: dict, category: str,
          garment_ids: list[str]) -> dict | None:
    """Make sure this cache row is completed. Returns the row, or None if the job
    was cancelled before anything was submitted for this step."""
    from tryon.vto_renderer import get_renderer, TryOnRenderError
    renderer = get_renderer()
    job_id, uid = str(job["job_id"]), str(job["user_id"])
    waiting_since = time.time()

    while True:
        status = _renew(job_id)
        row = _find(key)
        if row and row["status"] == "completed" and row.get("s3_key"):
            return row

        request_id = row.get("provider_request_id") if row else None
        if row and row["status"] == "processing" and request_id:
            # Submitted already (by us before a crash, or by another job): follow it.
            state, value = renderer.poll(request_id)
            if state == "done":
                try:
                    return _complete(row["id"], renderer.fetch(value), garment_ids)
                except Exception as exc:
                    if _is_network(exc) or "could not download" in str(exc):
                        # Paid for and ready at the provider: keep the row as it is and
                        # fetch it again after the blip (never render it a second time).
                        raise _Retry(f"store not finished: {exc}")
                    _row_failed(row["id"], f"store failed: {exc}")
                    if status == "cancelled":
                        return None
                    _fail(job_id, "provider_error", "Couldn't save the render. You weren't charged.")
            if state == "error":
                _row_failed(row["id"], str(value))
                if status == "cancelled":
                    return None
                _fail(job_id, "provider_error", "The render didn't work. You weren't charged.")
            # still queued / running at the provider
            submitted = _parse(row.get("provider_submitted_at")) or _now()
            if (_now() - submitted).total_seconds() > STEP_TIMEOUT:
                renderer.cancel(request_id)
                _row_failed(row["id"], "timeout")
                if status == "cancelled":
                    return None
                _fail(job_id, "timeout", "The render took too long. You weren't charged.")
            time.sleep(POLL_EVERY)
            continue

        if status != "running":
            return None                     # cancelled (or gone) before we paid for this step

        row_id = _claim(row, key, uid)
        if row_id is None:
            # Someone else holds the row without a provider request (old endpoint
            # rendering in-request): wait for it, but not forever.
            if time.time() - waiting_since > WAIT_FOR_OTHERS:
                _fail(job_id, "timeout", "The render took too long. You weren't charged.")
            time.sleep(POLL_EVERY)
            continue

        try:
            request_id = renderer.submit(person_url, garment["clean_image_url"], category)
        except (TryOnRenderError, Exception) as exc:
            _row_failed(row_id, f"submit failed: {exc}")
            if _not_sent(exc):
                raise _Retry(f"submit not sent: {exc}")
            _fail(job_id, "provider_error", "The render didn't start. You weren't charged.")
        supabase.table(RENDERS).update({"provider_request_id": request_id, "submitted_by_job": job_id,
                                        "provider_submitted_at": _iso(), "updated_at": _iso()})\
            .eq("id", row_id).execute()
        supabase.table(JOBS).update({"provider_started": True}).eq("job_id", job_id).execute()
        logger.info("render job %s submitted step key=%s request=%s", job_id, key.get("garment_id"), request_id)
        time.sleep(POLL_EVERY)


# ---------------------------------------------------------------- one job

def run_job(job_id: str) -> None:
    from pipelines.render_jobs_pipeline import step_keys
    t0 = time.time()
    try:
        job = _job(job_id)
        if not job or job["status"] not in ("running", "cancelled"):
            return
        if int(job.get("attempts") or 0) > MAX_ATTEMPTS:
            _fail(job_id, "provider_error", "The render couldn't finish. You weren't charged.")
        uid = str(job["user_id"])

        av = supabase.table("avatars").select("avatar_id, avatar_version, original_photo_url")\
            .eq("user_id", uid).limit(1).execute().data
        if not av or av[0]["avatar_version"] != job["avatar_version"]:
            _fail(job_id, "avatar_missing", "Your avatar changed. Render again.")
        avatar = av[0]

        steps = job["steps"] or []
        ids = [s["garment_id"] for s in steps]
        rows = supabase.table("closet_items").select("item_id, category, image_hash, clean_image_url")\
            .eq("user_id", uid).in_("item_id", ids).execute().data or []
        by_id = {str(r["item_id"]): r for r in rows}
        if any(i not in by_id for i in ids):
            _fail(job_id, "garment_missing", "A piece in this outfit was removed.")
        garments = [by_id[i] for i in ids]
        keys = step_keys(avatar, garments)

        result = None
        for i, step in enumerate(steps):
            person = avatar["original_photo_url"] if i == 0 else storage.get_url(result["s3_key"])
            result = _step(job, keys[i], person, garments[i], step["category"], ids[: i + 1])
            if result is None:                 # cancelled
                logger.info("render job %s stopped (cancelled) at step %s", job_id, i + 1)
                return
            supabase.table(JOBS).update({"steps_done": i + 1}).eq("job_id", job_id).execute()

        from pipelines.render_jobs_pipeline import period_now
        res = supabase.rpc("finish_render_job", {"p_job": job_id, "p_image": result["id"],
                                                 "p_period": period_now()}).execute()
        charged = res.data[0] if isinstance(res.data, list) else res.data
        if charged is not None and int(charged) >= 0:
            logger.info("render job %s succeeded in %.1fs, charged %s", job_id, time.time() - t0, charged)
        else:
            logger.info("render job %s finished after cancel in %.1fs (not charged)", job_id, time.time() - t0)
    except _Stop:
        pass
    except _Retry as exc:
        _park(job_id, str(exc))
    except Exception as exc:
        if _is_network(exc):
            _park(job_id, f"{type(exc).__name__}: {exc}")
            return
        logger.exception("render job %s crashed", job_id)
        try:
            _fail(job_id, "provider_error", f"Something went wrong ({type(exc).__name__}). You weren't charged.")
        except _Stop:
            pass
    finally:
        _running.discard(job_id)


# ---------------------------------------------------------------- loop

def _claim_jobs() -> list[dict]:
    res = supabase.rpc("claim_render_jobs", {"p_limit": _concurrency(),
                                             "p_lease_seconds": LEASE_SECONDS}).execute()
    return res.data or []


async def render_worker() -> None:
    """Background loop started with the app."""
    logger.info("render worker started (concurrency %s)", _concurrency())
    while True:
        try:
            for job in await asyncio.to_thread(_claim_jobs):
                jid = str(job["job_id"])
                if jid in _running:
                    continue
                _running.add(jid)
                asyncio.get_running_loop().run_in_executor(None, run_job, jid)
        except Exception:
            logger.exception("render worker pass failed")
        await asyncio.sleep(TICK)