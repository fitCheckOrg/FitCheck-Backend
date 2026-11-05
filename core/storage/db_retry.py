"""
Supabase connection-drop safety net (2026-11-05).

Problem: on a weak connection Supabase sometimes drops the socket mid-request
(httpx ReadError / ConnectError, "WinError 10054", "Server disconnected",
"getaddrinfo failed"). One dropped query used to fail the whole request with
500 INTERNAL_ERROR - and pipelines like E19 insights run several queries in a row.

Fix, in two parts:
1. install_db_retry() wraps every postgrest query's .execute(): on a dropped
   connection it waits RETRY_DELAY_S and tries ONCE more.
     * reads (GET/HEAD): retried on any connection error
     * writes (POST/PATCH/DELETE): retried ONLY when the request never left
       the machine (ConnectError / ConnectTimeout / PoolTimeout) - a write
       that may already have reached the database is never sent twice.
   (Newer postgrest versions already retry some GETs on their own; this adds
   one more attempt after a short pause and covers writes that never left.)
2. main.py turns any connection error that still escapes into
   503 DB_UNAVAILABLE + Retry-After, instead of 500 INTERNAL_ERROR.
"""

import logging
import time

import httpx

logger = logging.getLogger(__name__)

RETRY_DELAY_S = 0.5
NOT_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
MAYBE_SENT = (httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError,
              httpx.ReadTimeout, httpx.WriteTimeout)
DB_CONNECTION_ERRORS = NOT_SENT + MAYBE_SENT

_installed = False


def _method(builder) -> str | None:
    req = getattr(builder, "request", None)
    m = getattr(req, "http_method", None) or getattr(builder, "http_method", None)
    return str(m).upper() if m else None


def _wrap(execute):
    def execute_with_retry(self, *args, **kwargs):
        try:
            return execute(self, *args, **kwargs)
        except DB_CONNECTION_ERRORS as exc:
            method = _method(self)
            safe = isinstance(exc, NOT_SENT) or method in ("GET", "HEAD")
            if not safe:
                logger.warning("db write dropped (%s %s) - not retried: %s",
                               method, type(exc).__name__, exc)
                raise
            logger.warning("db connection dropped (%s %s) - retrying once: %s",
                           method, type(exc).__name__, exc)
            time.sleep(RETRY_DELAY_S)
            return execute(self, *args, **kwargs)
    execute_with_retry.__wrapped__ = execute
    return execute_with_retry


def install_db_retry() -> int:
    """Patch every postgrest sync request builder that defines execute(). Safe to call twice."""
    global _installed
    if _installed:
        return 0
    try:
        from postgrest._sync import request_builder as rb
    except Exception:
        logger.warning("db retry not installed: postgrest sync builders not found")
        return 0
    patched = 0
    for name in dir(rb):
        cls = getattr(rb, name)
        if isinstance(cls, type) and "execute" in cls.__dict__ \
                and not hasattr(cls.__dict__["execute"], "__wrapped__"):
            setattr(cls, "execute", _wrap(cls.__dict__["execute"]))
            patched += 1
    _installed = True
    logger.info("db retry installed on %s postgrest builders", patched)
    return patched