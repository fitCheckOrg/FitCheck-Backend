"""
ONB-4 (2026-11-12): keep the Node auth service's profile cache fresh.

The Node service (FitCheckBackend) caches each user's profiles row in Upstash
Redis for 15 minutes under the key  profile:{user_id}  and serves it from
GET /api/auth/me. When THIS service changes a profiles row (name, handle, bio,
style words, setup progress ...) we delete that key, so /auth/me shows the new
value on the next call instead of up to 15 minutes later.

Config (.env, the SAME two values the Node service uses):
    UPSTASH_REDIS_REST_URL
    UPSTASH_REDIS_REST_TOKEN
Not set -> nothing happens (logged once). Best effort: never raises, never
slows a request by more than CACHE_TIMEOUT seconds.
"""

import logging
import os

import requests

logger = logging.getLogger(__name__)

try:                                     # pick up .env if python-dotenv is installed
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

CACHE_TIMEOUT = 2
_warned = False


def invalidate_profile(user_id) -> None:
    """Delete the Node service's cached copy of this user's profile."""
    global _warned
    url = (os.getenv("UPSTASH_REDIS_REST_URL") or "").strip().rstrip("/")
    token = (os.getenv("UPSTASH_REDIS_REST_TOKEN") or "").strip()
    if not url or not token:
        if not _warned:
            logger.warning("profile cache not cleared: UPSTASH_REDIS_REST_URL / TOKEN not set")
            _warned = True
        return
    try:
        res = requests.post(f"{url}/del/profile:{user_id}",
                            headers={"Authorization": f"Bearer {token}"},
                            timeout=CACHE_TIMEOUT)
        if res.status_code >= 400:
            logger.warning("profile cache clear failed user=%s: HTTP %s", user_id, res.status_code)
    except Exception as exc:
        logger.warning("profile cache clear failed user=%s: %s", user_id, exc)