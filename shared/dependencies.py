import jwt
from fastapi import Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.concurrency import run_in_threadpool
from uuid import UUID
from core.config.settings import settings
from core.auth.jwt_verifier import verify_supabase_token, AuthUnavailable
from shared.logger import get_logger
from shared.exceptions import FitCheckException

logger = get_logger(__name__)

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    user_id: UUID | None = Query(None)
) -> UUID:
    """
    Single source of truth for identifying the authenticated user.
    Every route uses this instead of trusting a client-supplied user_id.

    Production: Authorization: Bearer <token> is required.
    The token is verified LOCALLY against Supabase's ES256 public keys
    (core/auth/jwt_verifier.py) — no network call per request.

        bad / expired / forged token     -> 401 INVALID_TOKEN
        signing keys can't be loaded     -> 503 AUTH_UNAVAILABLE (client retries,
                                            must NOT log the user out)

    Development (DEV_AUTH_BYPASS=true): falls back to ?user_id=
    if no Bearer token is provided — keeps curl/Postman
    testing easy without weakening production behavior.
    """
    if credentials and credentials.credentials:
        token = credentials.credentials.strip()
        if token:
            try:
                claims = await run_in_threadpool(verify_supabase_token, token)
            except AuthUnavailable as exc:
                logger.warning(f"[auth] verification unavailable: {exc}")
                raise FitCheckException(
                    "Authentication temporarily unavailable, please retry.",
                    code="AUTH_UNAVAILABLE", status_code=503
                )
            except jwt.PyJWTError:
                raise FitCheckException(
                    "Invalid or expired authentication token.",
                    code="INVALID_TOKEN", status_code=401
                )
            except Exception:
                logger.exception("[auth] unexpected verification error")
                raise FitCheckException(
                    "Authentication error.",
                    code="AUTH_ERROR", status_code=500
                )

            try:
                return UUID(claims["sub"])
            except (KeyError, ValueError):
                raise FitCheckException(
                    "Invalid authentication token.",
                    code="INVALID_TOKEN", status_code=401
                )

    if settings.DEV_AUTH_BYPASS and user_id is not None:
        logger.warning(f"DEV_AUTH_BYPASS active — trusting user_id={user_id} from query param")
        return user_id

    raise FitCheckException(
        "Authentication required. Please log in.",
        code="UNAUTHORIZED", status_code=401
    )