"""
Try-On v2 T2-1 B2 (2026-11-17): render jobs - the app's side.

POST /api/tryon/jobs                 {outfit_id} | {items}    -> 202 new job / 200 cached or existing job
GET  /api/tryon/jobs/{job_id}        -> job
GET  /api/tryon/jobs?active=true     -> {items: [job]}   (queued + running: recovery after a restart)
POST /api/tryon/jobs/{job_id}/cancel -> job
GET  /api/tryon/usage                -> {period, used, reserved, limit, remaining, resets_at, tier}   (B3)

job = {job_id, outfit_id, status queued|running|succeeded|failed|cancelled,
       progress {step, steps}, queue_position (queued only, 1 = next),
       rendered_slots, tray_slots, charged_generations, provider_started,
       failure {code, message} | null, image (image object) | null,
       created_at, finished_at}

B6 (2026-11-18): a job's image carries outdated ("garment_deleted" | "avatar_replaced")
and url null when the file was deleted.

Create decides, in order (design doc section 2):
    1. the finished render of exactly these pieces exists  -> 200, a job already 'succeeded',
       charged 0, no provider call
    2. the same render is already queued / running for me   -> 200, THAT job
    3. otherwise                                            -> 202, a queued job
The render itself happens in the background worker (pipelines/render_worker.py).

B3 (2026-11-17) - quotas, limits, ceiling (all numbers are .env settings):
    RENDER_CREATES_PER_HOUR  30   job creates per user per hour, cache hits included -> 429 RATE_LIMITED
    MAX_ACTIVE_PER_USER       2   queued + running per user                           -> 429 RATE_LIMITED
    FREE_RENDER_LIMIT        10   charged generations per UTC calendar month           -> 429 QUOTA_EXCEEDED
    PREMIUM_RENDER_LIMIT    100   same, tier premium / pro / enterprise
    RENDER_DAILY_CEILING    100   generations sent to the provider per UTC day, all users -> 503 RENDERS_PAUSED
  Order: hourly limit -> already rendered (free, opens even when paused / out of renders)
         -> same render on its way -> active / quota / ceiling (one locked DB step).
  B13 (2026-11-26): FASHN credits used up (or new renders paused by hand on the Developer
  screen) -> 503 RENDERS_PAUSED {retry_after} before anything is queued; already-rendered
  outfits still open. See pipelines/fashn_credits.py.
  "used" = charged generations this month; "reserved" = renders still expected by my
  queued / running jobs. A job is charged only when it SUCCEEDS (finish_render_job).
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from uuid import UUID

from core.storage.supabase_client import supabase
from shared.exceptions import FitCheckException, AvatarNotFoundError
from pipelines.tryon_slots import FASHN_CATEGORY

logger = logging.getLogger(__name__)

TABLE = "render_jobs"
MAX_ACTIVE_PER_USER = 2
ACTIVE = ("queued", "running")
JOB_FIELDS = ("job_id, user_id, outfit_id, avatar_id, avatar_version, steps, rendered_slots, tray_slots, "
              "status, failure_code, failure_message, steps_total, steps_done, image_id, "
              "charged_generations, provider_started, needs, created_at, finished_at")


PREMIUM_TIERS = {"premium", "pro", "enterprise"}


def _setting(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except ValueError:
        return default


def _bad(code: str, message: str, status: int = 400, headers: dict | None = None,
         details: dict | None = None) -> FitCheckException:
    exc = FitCheckException(message, code=code, status_code=status, headers=headers)
    exc.details = details or {}                 # extra fields in the error body (main.py)
    return exc


# ---------------------------------------------------------------- B3 period / limits

def period_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _resets_at() -> datetime:
    now = datetime.now(timezone.utc)
    year, month = (now.year + 1, 1) if now.month == 12 else (now.year, now.month + 1)
    return datetime(year, month, 1, tzinfo=timezone.utc)


def _seconds_to_midnight_utc() -> int:
    now = datetime.now(timezone.utc)
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(60, int((midnight - now).total_seconds()))


def _tier(uid: str) -> str:
    rows = supabase.table("profiles").select("tier").eq("id", uid).limit(1).execute().data
    return str((rows[0].get("tier") if rows else None) or "free").lower()


def _limit_for(tier: str) -> int:
    return _setting("PREMIUM_RENDER_LIMIT", 100) if tier in PREMIUM_TIERS \
        else _setting("FREE_RENDER_LIMIT", 10)


def get_usage(user_id) -> dict:
    uid = str(user_id)
    tier = _tier(uid)
    limit = _limit_for(tier)
    period = period_now()
    used = supabase.table("render_charges").select("render_id", count="exact")\
        .eq("user_id", uid).eq("period", period).limit(1).execute().count or 0
    active = supabase.table(TABLE).select("needs").eq("user_id", uid)\
        .in_("status", list(ACTIVE)).execute().data or []
    reserved = sum(int(r.get("needs") or 0) for r in active)
    return {"period": period, "used": used, "reserved": reserved, "limit": limit,
            "remaining": max(0, limit - used - reserved),
            "resets_at": _resets_at().isoformat(), "tier": tier}


def _check_hourly(uid: str) -> None:
    per_hour = _setting("RENDER_CREATES_PER_HOUR", 30)
    since = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    made = supabase.table(TABLE).select("job_id", count="exact").eq("user_id", uid)\
        .gte("created_at", since).limit(1).execute().count or 0
    if made >= per_hour:
        raise _bad("RATE_LIMITED", "You've asked for a lot of renders this hour. Try again a bit later.",
                   429, {"Retry-After": "600"}, {"retry_after": 600})
    day = supabase.table(TABLE).select("job_id", count="exact").eq("user_id", uid)\
        .gte("created_at", (datetime.now(timezone.utc) - timedelta(days=1)).isoformat())\
        .limit(1).execute().count or 0
    if day > 50:
        logger.warning("render abuse signal: user=%s created %s jobs in 24 h", uid, day)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _avatar(uid: str) -> dict:
    rows = supabase.table("avatars").select("avatar_id, avatar_version")\
        .eq("user_id", uid).limit(1).execute().data
    if not rows:
        raise AvatarNotFoundError()
    return rows[0]


# ---------------------------------------------------------------- shaping

def _queue_position(job: dict) -> int:
    ahead = supabase.table(TABLE).select("job_id", count="exact")\
        .eq("status", "queued").lt("created_at", job["created_at"]).limit(1).execute().count or 0
    return ahead + 1


def _shape(job: dict) -> dict:
    from pipelines.tryon_outfits_pipeline import _image
    image = None
    if job.get("image_id"):
        rows = supabase.table("tryon_renders").select("id, s3_key, status, outdated, file_deleted_at")\
            .eq("id", str(job["image_id"])).limit(1).execute().data
        if rows and rows[0].get("s3_key"):
            image = _image(rows[0], {"avatar_version": job["avatar_version"]},
                           [s["garment_id"] for s in (job.get("steps") or [])])
    status = job["status"]
    total = int(job.get("steps_total") or 0)
    done = int(job.get("steps_done") or 0)
    return {
        "job_id": str(job["job_id"]),
        "outfit_id": str(job["outfit_id"]),
        "status": status,
        "progress": {"step": min(done + 1, total) if status in ACTIVE else done, "steps": total},
        "queue_position": _queue_position(job) if status == "queued" else None,
        "rendered_slots": job.get("rendered_slots") or [],
        "tray_slots": job.get("tray_slots") or [],
        "charged_generations": int(job.get("charged_generations") or 0),
        "provider_started": bool(job.get("provider_started")),
        "failure": {"code": job["failure_code"], "message": job.get("failure_message")}
        if job.get("failure_code") else None,
        "image": image,
        "created_at": job["created_at"],
        "finished_at": job.get("finished_at"),
    }


def _get_row(uid: str, job_id) -> dict:
    try:
        jid = str(UUID(str(job_id)))
    except Exception:
        raise _bad("JOB_NOT_FOUND", "Render job not found.", 404)
    rows = supabase.table(TABLE).select(JOB_FIELDS).eq("job_id", jid).eq("user_id", uid)\
        .limit(1).execute().data
    if not rows:
        raise _bad("JOB_NOT_FOUND", "Render job not found.", 404)
    return rows[0]


# ---------------------------------------------------------------- create

def _plan_for(uid: str, outfit_id: str) -> tuple[dict, list[dict]]:
    """(outfit as the API returns it, rendered pieces in order with their garment rows)."""
    from pipelines.tryon_outfits_pipeline import get_outfit, _load_garments
    outfit = get_outfit(uid, outfit_id)
    rendered = [i for i in outfit["items"] if i["renders"]]
    garments = _load_garments(uid, [i["garment_id"] for i in rendered])
    pieces = [{**i, "garment": garments[i["garment_id"]]} for i in rendered if i["garment_id"] in garments]
    return outfit, pieces


def step_keys(avatar: dict, garments: list[dict]) -> list[dict]:
    """Cache identities of each step (shared with the single try-on cache)."""
    from pipelines.tryon_pipeline import _single_key, _chain_key
    keys = [_single_key(avatar, garments[0])]
    if len(garments) == 2:
        keys.append(_chain_key(avatar, garments[0], garments[1]))
    return keys


def create_job(user_id, outfit_id=None, items=None) -> tuple[dict, bool]:
    """Returns (job, created). created=False -> answer 200 (cached / existing job)."""
    from pipelines.tryon_outfits_pipeline import create_outfit
    from pipelines.tryon_pipeline import _find_row
    uid = str(user_id)
    _check_hourly(uid)                                  # B3: counts cache hits too
    if outfit_id is None:
        if not items:
            raise _bad("OUTFIT_EMPTY", "Send an outfit_id or the pieces to render.")
        outfit_id = create_outfit(uid, items)["outfit_id"]

    outfit, pieces = _plan_for(uid, str(outfit_id))
    if not pieces:
        raise _bad("NOTHING_TO_RENDER", "None of these pieces can be shown on the avatar.")
    avatar = _avatar(uid)
    garments = [p["garment"] for p in pieces]
    keys = step_keys(avatar, garments)
    signature = hashlib.sha256(json.dumps(keys[-1], sort_keys=True, default=str).encode()).hexdigest()
    steps = [{"garment_id": p["garment_id"], "slot": p["slot"], "category": FASHN_CATEGORY[p["slot"]]}
             for p in pieces]
    base = {
        "user_id": uid, "outfit_id": outfit["outfit_id"],
        "avatar_id": avatar["avatar_id"], "avatar_version": avatar["avatar_version"],
        "signature": signature, "steps": steps,
        "rendered_slots": outfit["rendered_slots"], "tray_slots": outfit["tray_slots"],
        "steps_total": len(steps),
    }

    # 1. Already rendered -> a finished job at once, nothing charged, no provider call.
    rows = [_find_row(k) for k in keys]
    done = [bool(r and r.get("status") == "completed" and r.get("s3_key")) for r in rows]
    final = rows[-1]
    if done[-1]:
        row = supabase.table(TABLE).insert({
            **base, "status": "succeeded", "steps_done": len(steps), "image_id": final["id"],
            "started_at": _now(), "finished_at": _now(),
        }).execute().data[0]
        logger.info("render job %s cached user=%s", row["job_id"], uid)
        return _shape(row), False

    # B13: no new paid renders while FASHN credits are used up / renders are paused by hand.
    from pipelines.fashn_credits import renders_blocked
    if renders_blocked():
        logger.warning("render job refused: new renders paused (credits / by hand) user=%s", uid)
        raise _bad("RENDERS_PAUSED", "Try-on is taking a short break. Your saved and earlier looks "
                   "still open; new renders will be back soon.", 503,
                   {"Retry-After": "900"}, {"retry_after": 900})

    # 2 + 3. Same render on its way -> that job; otherwise admit (active jobs, monthly
    # quota, daily ceiling) and create - one locked step in the database (B3).
    needs = len(keys) - (1 if len(keys) == 2 and done[0] else 0)
    tier = _tier(uid)
    limit = _limit_for(tier)
    res = supabase.rpc("create_render_job", {
        "p_user": uid,
        "p_job": {k: base[k] for k in ("outfit_id", "avatar_id", "avatar_version", "signature",
                                       "steps", "rendered_slots", "tray_slots", "steps_total")},
        "p_needs": needs, "p_limit": limit, "p_period": period_now(),
        "p_max_active": MAX_ACTIVE_PER_USER, "p_ceiling": _setting("RENDER_DAILY_CEILING", 100),
    }).execute()
    out = res.data[0] if isinstance(res.data, list) else res.data
    if not out.get("ok"):
        code = out.get("code")
        if code == "QUOTA_EXCEEDED":
            resets = _resets_at().isoformat()
            used, reserved = int(out.get("used") or 0), int(out.get("reserved") or 0)
            raise _bad("QUOTA_EXCEEDED",
                       f"This needs {needs} render{'s' if needs != 1 else ''} and you have "
                       f"{max(0, limit - used - reserved)} left this month.", 429,
                       details={"used": used, "reserved": reserved, "limit": limit,
                                "needed": needs, "resets_at": resets})
        if code == "RENDERS_PAUSED":
            wait = _seconds_to_midnight_utc()
            logger.warning("render ceiling reached - new renders paused (user=%s)", uid)
            raise _bad("RENDERS_PAUSED", "Try-on is very busy today. Your saved and earlier looks "
                       "still open; new renders are back tomorrow.", 503,
                       {"Retry-After": str(wait)}, {"retry_after": wait})
        raise _bad("RATE_LIMITED", "You already have renders on the way. Wait for one to finish.",
                   429, {"Retry-After": "20"}, {"retry_after": 20})
    job = _get_row(uid, out["job_id"])
    if out.get("created"):
        logger.info("render job %s queued user=%s steps=%s needs=%s", out["job_id"], uid, len(steps), needs)
    return _shape(job), bool(out.get("created"))


# ---------------------------------------------------------------- read / cancel

def get_job(user_id, job_id) -> dict:
    return _shape(_get_row(str(user_id), job_id))


def list_active(user_id) -> dict:
    rows = supabase.table(TABLE).select(JOB_FIELDS).eq("user_id", str(user_id))\
        .in_("status", list(ACTIVE)).order("created_at").execute().data or []
    return {"items": [_shape(r) for r in rows]}


def cancel_job(user_id, job_id) -> dict:
    """Queued -> cancelled at once. Running -> cancelled; if the provider already
    started, the worker still saves the result to the cache (never charged)."""
    uid = str(user_id)
    row = _get_row(uid, job_id)
    if row["status"] in ACTIVE:
        res = supabase.table(TABLE).update({
            "status": "cancelled", "failure_code": "cancelled",
            "failure_message": "Cancelled - not charged.", "finished_at": _now(),
        }).eq("job_id", row["job_id"]).in_("status", list(ACTIVE)).execute().data
        if res:
            row = res[0]
            logger.info("render job %s cancelled user=%s", row["job_id"], uid)
        else:
            row = _get_row(uid, job_id)                 # finished meanwhile
    return _shape(row)