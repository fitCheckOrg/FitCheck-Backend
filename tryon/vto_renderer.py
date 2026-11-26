"""
Try-on renderer: the ONLY module in the backend that knows which
try-on model/provider we use.

    renderer = get_renderer()
    png_bytes = renderer.render(person_image, garment_image, category="tops")   # blocking

Try-On v2 T2-1 B2 (2026-11-17) - the same provider, in steps, for the background
worker (pipelines/render_worker.py):

    request_id = renderer.submit(person_image, garment_image, category)   # returns at once
    state      = renderer.poll(request_id)    # ("queued", pos) | ("running", None)
                                              # | ("done", result_ref) | ("error", message)
    png_bytes  = renderer.fetch(result_ref)   # download with retries (already paid for)
    renderer.cancel(request_id)               # best effort

person_image / garment_image can be PIL Images, raw bytes, or http(s) URLs.
Swapping provider means adding a new TryOnRenderer subclass and changing
get_renderer(). Nothing else in the backend, and nothing in Flutter, changes.

B-TRY-1 (2026-11-03): every download (input photos and the finished render)
goes through _get_with_retry, so a short network drop (DNS getaddrinfo
failure, connection reset, timeout, 5xx) no longer wastes a paid render.

B2 speed fix (audit P4): the person photo is shrunk to PERSON_MAX_SIDE before it
is sent (FASHN's output is 864x1296; the full upload photo was up to 6000 px).

B5 (2026-11-18): avatar photos and renders are PRIVATE in S3. A link to our own
bucket is swapped for a fresh signed link before every download attempt, so the
stored plain URLs (avatars.original_photo_url) keep working after the bucket is locked.

B8 (2026-11-20): fal removed FASHN v1.6 ("This endpoint is deprecated and no longer
available"). The same model now runs on FASHN's own API (https://api.fashn.ai/v1,
model_name "tryon-v1.6", 1 credit per image, key FASHN_API_KEY in .env).
  * Inputs are sent as base64 and the output comes back as base64 (return_base64):
    FASHN keeps it 60 minutes, never on its CDN, and doesn't train on it.
  * poll() returns "fashn:<id>" instead of a URL; fetch() reads the image from the
    status endpoint (valid 60 min) with retries, so a finished render is never paid twice.
  * FASHN has no cancel endpoint: cancel() only logs (the render finishes and is cached).
  * name stays "fal-fashn-v1.6": same model, so existing cached renders keep matching.

B13 (2026-11-26): when FASHN refuses a render for credits (HTTP 402, or an error that
mentions credits), submit() tells pipelines/fashn_credits.py, which pauses new paid
renders and alerts the developer.

Smoke test (server must be running on :8000; costs 1 FASHN credit):
    python -m tryon.vto_renderer
"""

import base64
import io
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import requests
from PIL import Image

logger = logging.getLogger(__name__)

ImageInput = Union[Image.Image, bytes, str]

VALID_CATEGORIES = {"tops", "bottoms", "one-pieces", "auto"}

DOWNLOAD_ATTEMPTS = 4          # 1 try + 3 retries
DOWNLOAD_TIMEOUT = 30          # seconds per attempt
RETRY_WAITS = (2, 4, 8)        # seconds to wait before retry 1, 2, 3
PERSON_MAX_SIDE = 1536         # px, longest side of the person photo sent to the model

FASHN_API = "https://api.fashn.ai/v1"
FASHN_SUBMIT_TIMEOUT = 60      # seconds (the request carries both images)
FASHN_STATUS_TIMEOUT = 30      # seconds
RESULT_PREFIX = "fashn:"       # poll() -> "fashn:<id>"; fetch() reads it from /status
RENDER_TIMEOUT = 180           # seconds, blocking render() only
POLL_EVERY = 2                 # seconds, blocking render() only


class TryOnRenderError(Exception):
    """Provider failed, was rejected, or returned no image."""


# ---------------------------------------------------------------- helpers

def _ensure_fal_key():
    """Make sure FAL_KEY is in the environment, falling back to reading .env.
    (No longer used by the try-on renderer; kept for the T2-2 prototypes.)"""
    if os.environ.get("FAL_KEY"):
        return
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("FAL_KEY=") and "=" in line:
                os.environ["FAL_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
                return
    raise TryOnRenderError("FAL_KEY is not set (checked environment and .env)")


def _fashn_key() -> str:
    """FASHN_API_KEY from the environment, falling back to reading .env."""
    key = os.environ.get("FASHN_API_KEY", "").strip()
    if not key:
        env = Path(".env")
        if env.exists():
            for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if line.startswith("FASHN_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        raise TryOnRenderError("FASHN_API_KEY is not set (checked environment and .env)")
    os.environ["FASHN_API_KEY"] = key
    return key


def _fashn_error(body) -> str:
    """FASHN errors are {"error": "Name", "message": "..."} or {"error": {"name", "message"}}."""
    if not isinstance(body, dict):
        return str(body)[:300]
    err = body.get("error")
    if isinstance(err, dict):
        return f"{err.get('name') or 'error'}: {err.get('message') or ''}".rstrip(": ")
    if err:
        return f"{err}: {body.get('message') or ''}".rstrip(": ")
    return str(body)[:300]


def _get_with_retry(url: str, what: str) -> bytes:
    """
    GET an image with retries on network hiccups and 5xx.
    4xx (e.g. 403/404 - the file really isn't there) fails at once.
    """
    from core.storage.s3 import storage
    last_error = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            link = storage.url_for(url) if url.startswith(storage.base) else url   # B5
            r = requests.get(link, timeout=DOWNLOAD_TIMEOUT)
            if r.status_code == 200:
                if attempt > 1:
                    logger.info("%s download succeeded on attempt %d", what, attempt)
                return r.content
            if r.status_code < 500:
                raise TryOnRenderError(f"could not download {what} ({r.status_code})")
            last_error = f"HTTP {r.status_code}"
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < DOWNLOAD_ATTEMPTS:
            wait = RETRY_WAITS[min(attempt - 1, len(RETRY_WAITS) - 1)]
            logger.warning("%s download failed (attempt %d/%d): %s - retrying in %ds",
                           what, attempt, DOWNLOAD_ATTEMPTS, last_error, wait)
            time.sleep(wait)

    raise TryOnRenderError(
        f"could not download {what} after {DOWNLOAD_ATTEMPTS} attempts: {last_error}"
    )


def _to_pil(img: ImageInput, what: str = "image") -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, (bytes, bytearray)):
        return Image.open(io.BytesIO(img))
    if isinstance(img, str) and img.startswith("http"):
        return Image.open(io.BytesIO(_get_with_retry(img, what)))
    raise TypeError(f"Unsupported image input: {type(img)}")


def _flatten_on_white(img: Image.Image) -> Image.Image:
    """Our garment PNGs are transparent; models want a clean white background."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        return bg
    return img.convert("RGB")


def _shrink(img: Image.Image, max_side: int) -> Image.Image:
    """Never upscale; keeps the aspect ratio."""
    if max(img.size) <= max_side:
        return img
    out = img.copy()
    out.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return out


def _data_uri(img: Image.Image, fmt: str) -> str:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.convert("RGB").save(buf, format="JPEG", quality=92)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG")
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _decode_output(out: str) -> bytes:
    """A FASHN output: a data URI / plain base64 (return_base64) or, defensively, a URL."""
    if out.startswith("data:"):
        return base64.b64decode(out.split(",", 1)[1])
    if out.startswith("http"):
        return _get_with_retry(out, "try-on result")
    return base64.b64decode(out)


# ---------------------------------------------------------------- interface

class TryOnRenderer(ABC):
    """Person photo + garment photo -> PNG bytes of the person wearing it."""

    #: short id stored with every cached render, so we know what made it
    name: str = "base"

    @abstractmethod
    def render(self, person_image: ImageInput, garment_image: ImageInput,
               category: str = "tops") -> bytes:
        ...

    # Step API for the background worker (T2-1 B2).
    @abstractmethod
    def submit(self, person_image: ImageInput, garment_image: ImageInput,
               category: str = "tops") -> str:
        ...

    @abstractmethod
    def poll(self, request_id: str) -> tuple[str, object]:
        ...

    def fetch(self, result_url: str) -> bytes:
        return _get_with_retry(result_url, "try-on result")

    def cancel(self, request_id: str) -> None:
        return None


# ---------------------------------------------------------------- FASHN (own API)

class FashnApiRenderer(TryOnRenderer):
    """FASHN v1.6 on FASHN's own API. One sample, fixed seed, quality mode, base64 in/out."""

    MODEL = "tryon-v1.6"
    ENDPOINT = f"{FASHN_API}/run ({MODEL})"
    name = "fal-fashn-v1.6"     # same model as before B8; kept so cached renders keep matching

    def __init__(self, seed: int = 42, mode: str = "quality"):
        self.seed = seed
        self.mode = mode

    @staticmethod
    def _headers() -> dict:
        return {"Authorization": f"Bearer {_fashn_key()}", "Content-Type": "application/json"}

    def _inputs(self, person_image: ImageInput, garment_image: ImageInput, category: str) -> dict:
        if category not in VALID_CATEGORIES:
            raise ValueError(f"category must be one of {sorted(VALID_CATEGORIES)}, got {category!r}")
        person = _shrink(_to_pil(person_image, "person photo").convert("RGB"), PERSON_MAX_SIDE)
        garment = _flatten_on_white(_to_pil(garment_image, "garment photo"))
        return {
            "model_image": _data_uri(person, "JPEG"),
            "garment_image": _data_uri(garment, "PNG"),
            "category": category,
            "garment_photo_type": "flat-lay",
            "mode": self.mode,
            "num_samples": 1,
            "seed": self.seed,
            "output_format": "png",
            "return_base64": True,
        }

    def _status(self, request_id: str) -> tuple[int, dict]:
        r = requests.get(f"{FASHN_API}/status/{request_id}", headers=self._headers(),
                         timeout=FASHN_STATUS_TIMEOUT)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, {}

    # -- steps (background worker)

    def submit(self, person_image: ImageInput, garment_image: ImageInput,
               category: str = "tops") -> str:
        body = {"model_name": self.MODEL, "inputs": self._inputs(person_image, garment_image, category)}
        try:
            r = requests.post(f"{FASHN_API}/run", json=body, headers=self._headers(),
                              timeout=FASHN_SUBMIT_TIMEOUT)
        except requests.RequestException as exc:
            raise TryOnRenderError(f"{self.name} submit failed: {exc}") from exc
        try:
            data = r.json()
        except ValueError:
            data = {}
        if r.status_code != 200 or not data.get("id"):
            detail = _fashn_error(data) if data else r.text[:300]
            if r.status_code == 402 or "credit" in str(detail).lower():
                try:
                    from pipelines.fashn_credits import note_out_of_credits
                    note_out_of_credits(f"HTTP {r.status_code}: {detail}")
                except Exception:
                    logger.exception("couldn't report the out-of-credits refusal")
            raise TryOnRenderError(f"{self.name} submit refused ({r.status_code}): {detail}")
        return data["id"]

    def poll(self, request_id: str) -> tuple[str, object]:
        try:
            code, data = self._status(request_id)
        except requests.RequestException as exc:
            # A network blip while asking is not a failed render: ask again next time.
            logger.warning("%s status check failed for %s: %s", self.name, request_id, exc)
            return "running", None
        except TryOnRenderError as exc:
            return "error", str(exc)
        if code == 429 or code >= 500:
            logger.warning("%s status check for %s got HTTP %s - asking again", self.name, request_id, code)
            return "running", None
        if code != 200:
            return "error", f"status check refused ({code}): {_fashn_error(data)}"
        status = data.get("status")
        if status == "completed":
            if not (data.get("output") or []):
                return "error", "no image in the result"
            return "done", f"{RESULT_PREFIX}{request_id}"
        if status in ("failed", "canceled", "time_out"):
            return "error", _fashn_error(data) if data.get("error") else str(status)
        return "running", None          # starting / in_queue / processing

    def fetch(self, result_url: str) -> bytes:
        if not result_url.startswith(RESULT_PREFIX):
            return super().fetch(result_url)
        request_id = result_url[len(RESULT_PREFIX):]
        last_error = None
        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            try:
                code, data = self._status(request_id)
                if code == 200:
                    out = (data.get("output") or [None])[0]
                    if data.get("status") != "completed" or not out or "_expired" in str(out)[:20]:
                        raise TryOnRenderError(
                            f"result not available ({data.get('status')}); FASHN keeps base64 results 60 min")
                    if attempt > 1:
                        logger.info("try-on result read on attempt %d", attempt)
                    return _decode_output(out)
                if code < 500 and code != 429:
                    raise TryOnRenderError(f"could not read the result ({code}): {_fashn_error(data)}")
                last_error = f"HTTP {code}"
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"

            if attempt < DOWNLOAD_ATTEMPTS:
                wait = RETRY_WAITS[min(attempt - 1, len(RETRY_WAITS) - 1)]
                logger.warning("try-on result read failed (attempt %d/%d): %s - retrying in %ds",
                               attempt, DOWNLOAD_ATTEMPTS, last_error, wait)
                time.sleep(wait)

        raise TryOnRenderError(
            f"could not read the try-on result after {DOWNLOAD_ATTEMPTS} attempts: {last_error}"
        )

    def cancel(self, request_id: str) -> None:
        # FASHN's API has no cancel endpoint. The render finishes on their side and the
        # worker caches it (never charged to a user who cancelled).
        logger.info("%s cancel %s: no cancel endpoint at FASHN - left to finish", self.name, request_id)

    # -- blocking (old /tryon endpoints)

    def render(self, person_image: ImageInput, garment_image: ImageInput,
               category: str = "tops") -> bytes:
        request_id = self.submit(person_image, garment_image, category)
        t0 = time.time()
        while time.time() - t0 < RENDER_TIMEOUT:
            time.sleep(POLL_EVERY)
            state, value = self.poll(request_id)
            if state == "done":
                # Already paid for - fetch() retries instead of failing.
                return self.fetch(value)
            if state == "error":
                raise TryOnRenderError(f"{self.name}: {value}")
        raise TryOnRenderError(f"{self.name}: no result after {RENDER_TIMEOUT}s")


# ---------------------------------------------------------------- factory

def get_renderer() -> TryOnRenderer:
    """The one place that decides which provider the backend uses."""
    return FashnApiRenderer()


# ---------------------------------------------------------------- smoke test

if __name__ == "__main__":
    import time as _time

    USER = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"          # Keji
    GARMENT = "d20577a1-01e9-4d6f-87a1-5ca89c3badba"       # black tee
    API = "http://127.0.0.1:8000/api"

    def _data(url):
        body = requests.get(url, timeout=60).json()
        return body.get("data", body)

    avatar = _data(f"{API}/avatar/me?user_id={USER}")
    garment = _data(f"{API}/wardrobe/{GARMENT}?user_id={USER}")

    renderer = get_renderer()
    t0 = _time.time()
    png = renderer.render(avatar["original_photo_url"], garment["clean_image_url"], "tops")
    out = Path("outputs/vto_spike/renderer_smoke.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png)
    print(f"OK  {renderer.name} via {renderer.ENDPOINT}  {len(png)//1024} KB  {_time.time()-t0:.1f}s  -> {out}")