"""
Try-On Studio B13 (2026-11-26): FASHN credit monitoring - an operational alert for the
developer. Nothing here is ever shown to ordinary FitCheck users (they only see the
existing 503 RENDERS_PAUSED when new renders are stopped).

Balance
  * FASHN's own balance (GET https://api.fashn.ai/v1/credits) every 15 minutes
    (every 2 minutes while exhausted), and on demand from the Developer screen.
  * Our own estimate between checks: last real balance minus every step sent to FASHN
    since then (tryon_renders.provider_submitted_at; 1 credit per step). Used when the
    balance check fails - a failing check is reported separately from a low balance.

Levels (percent of the "full" amount)
    ok > 25 %  ·  low <= 25 %  ·  critical <= 10 %  ·  near_empty <= 5 %  ·  exhausted = 0
  "Full" = FASHN_CREDIT_FULL in .env if set, otherwise the balance right after the last
  top-up (the server sees the balance go up). A top-up resets the alerts.

Alerts (log + email, pipelines/ops_email.py)
  * One alert each time the balance crosses into a WORSE level - never repeated per check.
  * Balance check failing for an hour -> one "check failing" alert.
  * Every alert is stored (provider_credit_alerts) with time, balance and level.
  * Logs carry balances only, never the API key or account details.

Exhausted (balance 0, or FASHN refuses a render for credits)
  * New paid renders stop at once: POST /tryon/jobs -> 503 RENDERS_PAUSED.
    Already-rendered outfits still open (free). They resume by themselves after a top-up.
  * The Developer screen can also pause new renders by hand (a kill switch).
"""

import asyncio
import logging
import os
from datetime import datetime, timezone

import requests

from core.storage.supabase_client import supabase
from pipelines.ops_email import send_ops_email, email_configured

logger = logging.getLogger(__name__)

PROVIDER = "fashn"
STATE = "provider_credit_state"
ALERTS = "provider_credit_alerts"
DEFAULT_URL = "https://api.fashn.ai/v1/credits"
CHECK_EVERY = 15 * 60
CHECK_EVERY_EXHAUSTED = 2 * 60
FAIL_ALERT_AFTER = 60 * 60
LEVELS = ["ok", "low", "critical", "near_empty", "exhausted"]
THRESHOLDS = [("near_empty", 5), ("critical", 10), ("low", 25)]      # percent, checked in order
TOPUP_LINK = "app.fashn.ai -> Developer API -> API Billing"


class CreditCheckError(Exception):
    """The balance couldn't be read (not the same as a low balance)."""


# ---------------------------------------------------------------- small helpers

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None = None) -> str:
    return (dt or _now()).isoformat()


def _parse(ts) -> datetime | None:
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00")) if ts else None


def _num(v) -> float | None:
    try:
        return None if v is None or isinstance(v, bool) else float(v)
    except (TypeError, ValueError):
        return None


def _fmt(n: float | None) -> str:
    if n is None:
        return "?"
    return str(int(n)) if float(n).is_integer() else f"{n:.2f}"


def _state() -> dict:
    rows = supabase.table(STATE).select("*").eq("provider", PROVIDER).limit(1).execute().data
    return rows[0] if rows else {"provider": PROVIDER}


def _save(fields: dict) -> None:
    supabase.table(STATE).upsert({"provider": PROVIDER, **fields, "updated_at": _iso()}).execute()


def _full_setting() -> float | None:
    n = _num(os.getenv("FASHN_CREDIT_FULL"))
    return n if n and n > 0 else None


def _percent(balance: float | None, reference: float | None) -> float | None:
    if balance is None or not reference or reference <= 0:
        return None
    return round(max(balance, 0) / reference * 100, 1)


def level_for(balance: float | None, reference: float | None) -> str | None:
    if balance is None:
        return None
    if balance <= 0:
        return "exhausted"
    pct = _percent(balance, reference)
    if pct is None:
        return "ok"
    for name, limit in THRESHOLDS:
        if pct <= limit:
            return name
    return "ok"


# ---------------------------------------------------------------- FASHN balance

def _extract_total(data) -> float | None:
    for path in (("credits", "total"), ("total",), ("credits",), ("balance",), ("data", "total")):
        cur = data
        for key in path:
            cur = cur.get(key) if isinstance(cur, dict) else None
        n = _num(cur)
        if n is not None:
            return n
    return None


def fetch_balance() -> float:
    from tryon.vto_renderer import _fashn_key, TryOnRenderError
    try:
        key = _fashn_key()
    except TryOnRenderError:
        raise CreditCheckError("FASHN_API_KEY is not set")
    url = (os.getenv("FASHN_CREDITS_URL") or DEFAULT_URL).strip()
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=20)
    except requests.RequestException as exc:
        raise CreditCheckError(f"network: {type(exc).__name__}")
    if r.status_code != 200:
        raise CreditCheckError(f"HTTP {r.status_code}")
    try:
        data = r.json()
    except ValueError:
        raise CreditCheckError("answer is not JSON")
    total = _extract_total(data)
    if total is None:
        keys = sorted(data)[:8] if isinstance(data, dict) else type(data).__name__
        raise CreditCheckError(f"unexpected answer (keys: {keys})")
    return total


def _sent_since(ts) -> int:
    if not ts:
        return 0
    return supabase.table("tryon_renders").select("id", count="exact")\
        .gt("provider_submitted_at", str(ts)).limit(1).execute().count or 0


def estimate(st: dict) -> float | None:
    bal = _num(st.get("balance"))
    if bal is None:
        return None
    try:
        return bal - _sent_since(st.get("checked_at"))
    except Exception:
        return bal


# ---------------------------------------------------------------- alerts

def _alert(level: str, subject: str, body: str, balance, reference, source: str) -> None:
    result = send_ops_email(subject, body)
    try:
        supabase.table(ALERTS).insert({
            "provider": PROVIDER, "level": level, "balance": balance, "reference": reference,
            "percent": _percent(_num(balance), _num(reference)), "source": source, "email": result,
        }).execute()
    except Exception:
        logger.warning("fashn credits: couldn't store the alert record")
    logger.warning("fashn credits ALERT %s: %s (email: %s)", level, subject, result)


def _maybe_alert(st: dict, level: str, balance, reference, source: str) -> None:
    """One alert per crossing into a worse level; a better level (top-up) re-arms them."""
    last = st.get("last_alert_level") or "ok"
    if level not in LEVELS or last not in LEVELS:
        return
    if LEVELS.index(level) > LEVELS.index(last):
        pct = _percent(_num(balance), _num(reference))
        if level == "exhausted":
            subject = "FASHN credits are used up - new renders are paused"
        else:
            subject = f"FASHN credits {level.replace('_', ' ')}: {_fmt(balance)} left"
        body = (
            f"Level: {level} (low <= 25 %, critical <= 10 %, near empty <= 5 %).\n"
            f"Balance: {_fmt(balance)} credits"
            + (f" of {_fmt(reference)} ({pct} %)" if pct is not None else "") + ".\n"
            f"Source: {source}.\n"
            f"Time: {_now().strftime('%Y-%m-%d %H:%M UTC')}.\n\n"
            + ("New paid renders are paused; already-rendered outfits still open. "
               "They resume by themselves after a top-up.\n" if level == "exhausted" else
               "Renders pause automatically at 0.\n")
            + f"Top up: {TOPUP_LINK}.\n"
        )
        _alert(level, subject, body, balance, reference, source)
        _save({"last_alert_level": level, "last_alert_at": _iso(), "last_alert_balance": balance})
    elif LEVELS.index(level) < LEVELS.index(last):
        _save({"last_alert_level": level})
        logger.info("fashn credits: back to %s (was %s) - alerts re-armed", level, last)


# ---------------------------------------------------------------- checks

def run_check(reason: str = "scheduled") -> dict:
    st = _state()
    prev = _num(st.get("balance"))
    try:
        balance = fetch_balance()
    except CreditCheckError as exc:
        return _check_failed(st, str(exc), reason)

    auto_ref = _num(st.get("auto_reference"))
    if auto_ref is None or prev is None or balance > prev:
        auto_ref = balance                          # first check, or a top-up
    reference = _full_setting() or auto_ref
    level = level_for(balance, reference)
    exhausted = level == "exhausted"
    if st.get("exhausted") and not exhausted:
        logger.info("fashn credits: balance back above 0 - new renders resume")
    _save({"balance": balance, "reference": reference, "auto_reference": auto_ref,
           "level": level, "checked_at": _iso(),
           "check_ok": True, "check_error": None, "failing_since": None,
           "fail_alert_sent": False, "exhausted": exhausted})
    logger.info("fashn credits (%s): %s of %s (%s %%) - %s", reason, _fmt(balance), _fmt(reference),
                _percent(balance, reference), level)
    _maybe_alert(st, level, balance, reference, "FASHN balance")
    return status()


def _check_failed(st: dict, message: str, reason: str) -> dict:
    failing_since = _parse(st.get("failing_since")) or _now()
    est = estimate(st)
    reference = _num(st.get("reference"))
    est_level = level_for(est, reference)
    fields = {"check_ok": False, "check_error": message[:200], "failing_since": _iso(failing_since)}
    if est_level:
        fields["level"] = est_level
    _save(fields)
    logger.warning("fashn credits (%s): balance check FAILED (%s) - estimate %s", reason, message, _fmt(est))
    if est_level:                                   # estimate never pauses renders by itself
        _maybe_alert(st, est_level, est, reference, "our estimate (balance check failing)")
    if (_now() - failing_since).total_seconds() >= FAIL_ALERT_AFTER and not st.get("fail_alert_sent"):
        _alert("check_failing", "FASHN balance check is failing",
               f"The FASHN balance couldn't be read for over an hour ({message}).\n"
               f"Estimated balance from our own render records: {_fmt(est)} credits.\n"
               "This is a monitoring problem, not a low balance. Renders keep running.\n",
               est, reference, "check failing")
        _save({"fail_alert_sent": True})
    return status()


def note_out_of_credits(detail: str) -> None:
    """Called by the renderer when FASHN refuses a render for credits. Never raises."""
    try:
        st = _state()
        if st.get("exhausted"):
            return
        _save({"exhausted": True, "level": "exhausted"})
        logger.error("fashn credits: FASHN refused a render for credits (%s) - new renders paused",
                     detail[:120])
        _maybe_alert(st, "exhausted", _num(st.get("balance")), _num(st.get("reference")),
                     "FASHN refused a render")
    except Exception:
        logger.exception("fashn credits: couldn't record the out-of-credits refusal")


# ---------------------------------------------------------------- render gate + status

def renders_blocked() -> str | None:
    """'paused' (by hand) / 'exhausted' / None. Never blocks because of our own failure."""
    try:
        st = _state()
    except Exception:
        return None
    if st.get("paused_manually"):
        return "paused"
    if st.get("exhausted"):
        return "exhausted"
    return None


def set_manual_pause(paused: bool) -> dict:
    _save({"paused_manually": bool(paused)})
    logger.warning("fashn credits: new renders %s by hand", "PAUSED" if paused else "resumed")
    return status()


def status() -> dict:
    st = _state()
    bal, ref = _num(st.get("balance")), _num(st.get("reference"))
    recent = []
    try:
        recent = supabase.table(ALERTS).select("level, balance, percent, source, email, created_at")\
            .eq("provider", PROVIDER).order("created_at", desc=True).limit(5).execute().data or []
    except Exception:
        pass
    return {
        "provider": PROVIDER,
        "balance": bal,
        "balance_estimate": estimate(st),
        "reference": ref,
        "reference_source": "FASHN_CREDIT_FULL" if _full_setting() else "balance after the last top-up",
        "percent": _percent(bal, ref),
        "level": st.get("level") or "unknown",
        "thresholds_percent": {"low": 25, "critical": 10, "near_empty": 5},
        "exhausted": bool(st.get("exhausted")),
        "paused_manually": bool(st.get("paused_manually")),
        "renders_blocked": bool(st.get("exhausted") or st.get("paused_manually")),
        "checked_at": st.get("checked_at"),
        "check_ok": st.get("check_ok"),
        "check_error": st.get("check_error"),
        "failing_since": st.get("failing_since"),
        "last_alert": {"level": st.get("last_alert_level"), "at": st.get("last_alert_at"),
                       "balance": _num(st.get("last_alert_balance"))} if st.get("last_alert_at") else None,
        "recent_alerts": recent,
        "email_configured": email_configured(),
    }


async def credit_monitor() -> None:
    """Background loop started with the app."""
    logger.info("fashn credit monitor started (every %s min)", CHECK_EVERY // 60)
    await asyncio.sleep(5)
    while True:
        wait = CHECK_EVERY
        try:
            st = await asyncio.to_thread(run_check)
            if st.get("exhausted"):
                wait = CHECK_EVERY_EXHAUSTED
        except Exception:
            logger.exception("fashn credit check crashed")
        await asyncio.sleep(wait)