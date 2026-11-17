"""
Try-On v2 T2-1 B2 (2026-11-17): render jobs - the app's side.

POST /api/tryon/jobs                 {outfit_id} | {items}    -> 202 new job / 200 cached or existing job
GET  /api/tryon/jobs/{job_id}        -> job
GET  /api/tryon/jobs?active=true     -> {items: [job]}   (queued + running: recovery after a restart)
POST /api/tryon/jobs/{job_id}/cancel -> job

job = {job_id, outfit_id, status queued|running|succeeded|failed|cancelled,
       progress {step, steps}, queue_position (queued only, 1 = next),
       rendered_slots, tray_slots, charged_generations, provider_started,
       failure {code, message} | null, image (image object) | null,
       created_at, finished_at}

Create decides, in order (design doc section 2):
    1. the finished render of exactly these pieces exists  -> 200, a job already 'succeeded',
       charged 0, no provider call
    2. the same render is already queued / running for me   -> 200, THAT job
    3. otherwise                                            -> 202, a queued job
The render itself happens in the background worker (pipelines/render_worker.py).
Quotas / ceiling come in B3, push in B4.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from uuid import UUID

from postgrest.exceptions import APIError

from core.storage.supabase_client import supabase
from shared.exceptions import FitCheckException, AvatarNotFoundError
from pipelines.tryon_slots import FASHN_CATEGORY

logger = logging.getLogger(__name__)

TABLE = "render_jobs"
MAX_ACTIVE_PER_USER = 2
ACTIVE = ("queued", "running")
JOB_FIELDS = ("job_id, user_id, outfit_id, avatar_id, avatar_version, steps, rendered_slots, tray_slots, "
              "status, failure_code, failure_message, steps_total, steps_done, image_id, "
              "charged_generations, provider_started, created_at, finished_at")


def _bad(code: str, message: str, status: int = 400, headers: dict | None = None) -> FitCheckException:
    return FitCheckException(message, code=code, status_code=status, headers=headers)


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
        rows = supabase.table("tryon_renders").select("id, s3_key, status")\
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
    final = _find_row(keys[-1])
    if final and final.get("status") == "completed" and final.get("s3_key"):
        row = supabase.table(TABLE).insert({
            **base, "status": "succeeded", "steps_done": len(steps), "image_id": final["id"],
            "started_at": _now(), "finished_at": _now(),
        }).execute().data[0]
        logger.info("render job %s cached user=%s", row["job_id"], uid)
        return _shape(row), False

    # 2. Same render already on its way -> that job.
    existing = supabase.table(TABLE).select(JOB_FIELDS).eq("user_id", uid)\
        .eq("signature", signature).in_("status", list(ACTIVE)).limit(1).execute().data
    if existing:
        return _shape(existing[0]), False

    # 3. New job (limits: B2 keeps 2 active per user; quotas / ceiling are B3).
    active = supabase.table(TABLE).select("job_id", count="exact").eq("user_id", uid)\
        .in_("status", list(ACTIVE)).limit(1).execute().count or 0
    if active >= MAX_ACTIVE_PER_USER:
        raise _bad("RATE_LIMITED", "You already have renders on the way. Wait for one to finish.",
                   429, {"Retry-After": "20"})
    try:
        row = supabase.table(TABLE).insert({**base, "status": "queued"}).execute().data[0]
    except APIError as exc:
        if getattr(exc, "code", None) == "23505":      # a parallel tap made it first
            existing = supabase.table(TABLE).select(JOB_FIELDS).eq("user_id", uid)\
                .eq("signature", signature).in_("status", list(ACTIVE)).limit(1).execute().data
            if existing:
                return _shape(existing[0]), False
        raise
    logger.info("render job %s queued user=%s steps=%s", row["job_id"], uid, len(steps))
    return _shape(row), True


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