"""
Try-On v2 T2-1 B5 (2026-11-18): every JSON response leaves with working links.

Any link to a PRIVATE file of our bucket inside a JSON response (renders, avatar
photos, garment originals - stored in the database as plain URLs) is swapped for
a presigned link (1 hour, see core/storage/s3.py). Public files (clean cut-outs,
profile photos) are left as they are. This covers every endpoint at once, so no
response can hand out a dead public link to a private file once the bucket is locked.

Image objects ({key, url, expires_at}) are built with storage.signed() and already
carry a signed url; the app re-signs with POST /api/media/sign when one expires.
"""

import re

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from core.storage.s3 import storage

_LINK = re.compile(re.escape(storage.base) + r'([^"\s?\\]+)(?:\?[^"\s\\]*)?')


def _swap(match: re.Match) -> str:
    key = match.group(1)
    if not storage.is_private(key):
        return match.group(0)
    return storage.presign(key)[0]


def sign_links(text: str) -> str:
    return _LINK.sub(_swap, text) if storage.base in text else text


class SignPrivateLinks(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if "application/json" not in (response.headers.get("content-type") or ""):
            return response
        body = b"".join([chunk async for chunk in response.body_iterator])
        text = body.decode("utf-8")
        signed = sign_links(text)
        headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
        return Response(content=signed.encode("utf-8"), status_code=response.status_code,
                        headers=headers, media_type="application/json")