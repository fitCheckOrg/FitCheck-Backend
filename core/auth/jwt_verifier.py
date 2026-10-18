"""
core/auth/jwt_verifier.py

Local verification of Supabase access tokens (ES256) against the project's
public keys (JWKS). No per-request call to Supabase.

    verify_supabase_token(token) -> claims dict (sub, email, role, ...)
        raises jwt.PyJWTError     -> token is bad          -> 401
        raises AuthUnavailable    -> keys can't be loaded  -> 503

Keys are fetched once at startup (background thread), refreshed in the
background every hour, and re-fetched immediately when a token arrives with
an unknown key id (Supabase key rotation), at most once every 30 s.
Mirrors FitCheckBackend/src/config/jwtVerifier.js.
"""

import logging
import os
import threading
import time

import jwt
import requests

logger = logging.getLogger(__name__)

BACKGROUND_REFRESH_S = 60 * 60
MIN_REFETCH_GAP_S = 30
FETCH_TIMEOUT_S = 5
LEEWAY_S = 30


class AuthUnavailable(Exception):
    """Signing keys could not be loaded (network / Supabase problem)."""


_lock = threading.Lock()
_keys: dict[str, jwt.PyJWK] = {}   # kid -> key
_last_success = 0.0
_last_attempt = 0.0


def _supabase_url() -> str:
    url = os.environ.get("SUPABASE_URL")
    if not url:
        try:
            from core.config.settings import settings
            url = getattr(settings, "SUPABASE_URL", None)
        except Exception:
            url = None
    if not url:
        raise RuntimeError("SUPABASE_URL is not set (env or settings)")
    return str(url).rstrip("/")


def _fetch_jwks() -> None:
    """Fetch and install the key set. Raises on any failure."""
    global _keys, _last_success, _last_attempt
    with _lock:
        _last_attempt = time.time()
        resp = requests.get(
            f"{_supabase_url()}/auth/v1/.well-known/jwks.json", timeout=FETCH_TIMEOUT_S
        )
        resp.raise_for_status()
        jwk_set = jwt.PyJWKSet.from_dict(resp.json())
        keys = {k.key_id: k for k in jwk_set.keys if k.key_id}
        if not keys:
            raise RuntimeError("JWKS response contains no keys")
        _keys = keys
        _last_success = time.time()
    logger.info("[auth] JWKS loaded (%d key(s))", len(keys))


def _refresh_in_background() -> None:
    def run():
        try:
            _fetch_jwks()
        except Exception as exc:
            logger.warning("[auth] background JWKS refresh failed (keeping cached keys): %s", exc)
    threading.Thread(target=run, daemon=True, name="jwks-refresh").start()


def _ensure_keys() -> None:
    if not _keys:
        try:
            _fetch_jwks()
        except Exception as exc:
            raise AuthUnavailable(f"Could not load signing keys: {exc}") from exc
    elif time.time() - _last_success > BACKGROUND_REFRESH_S and not _lock.locked():
        _refresh_in_background()


def _decode(token: str, key: jwt.PyJWK) -> dict:
    return jwt.decode(
        token,
        key=key.key,
        algorithms=["ES256"],
        audience="authenticated",
        issuer=f"{_supabase_url()}/auth/v1",
        leeway=LEEWAY_S,
        options={"require": ["exp", "sub", "iss", "aud"]},
    )


def verify_supabase_token(token: str) -> dict:
    """Blocking only when keys must be fetched; call via run_in_threadpool."""
    _ensure_keys()

    kid = jwt.get_unverified_header(token).get("kid")  # raises PyJWTError if malformed
    key = _keys.get(kid)

    if key is None:
        # Unknown key id: Supabase may have rotated keys. Re-fetch once.
        if time.time() - _last_attempt > MIN_REFETCH_GAP_S:
            try:
                _fetch_jwks()
            except Exception as exc:
                raise AuthUnavailable(f"Could not refresh signing keys: {exc}") from exc
            key = _keys.get(kid)
        if key is None:
            raise jwt.InvalidTokenError(f"No signing key matches kid={kid!r}")

    return _decode(token, key)


def _warm_up() -> None:
    try:
        _fetch_jwks()
    except Exception as exc:
        logger.warning("[auth] JWKS warm-up failed, will retry on first request: %s", exc)


# Load keys as soon as the app imports this module, without blocking startup.
threading.Thread(target=_warm_up, daemon=True, name="jwks-warmup").start()