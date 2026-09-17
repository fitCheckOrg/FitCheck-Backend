import time

import requests

from core.config.settings import settings

from shared.logger import get_logger

from shared.exceptions import BackgroundRemovalError

logger = get_logger(__name__)

_session = requests.Session()

MAX_RETRIES = 2

RETRY_BACKOFF = 1.5  # seconds, multiplied per attempt

def remove_background(image_bytes: bytes) -> bytes:

    """

    Sends image to remove.bg API.

    Returns clean image bytes with background removed.

    """

    last_exc = None

    for attempt in range(1, MAX_RETRIES + 2):  # e.g. 3 total tries

        start = time.perf_counter()

        try:

            logger.info(f"Sending image to remove.bg (attempt {attempt})")

            response = _session.post(

                "https://api.remove.bg/v1.0/removebg",

                files={"image_file": ("upload.png", image_bytes, "image/png")},

                data={"size": "auto", "crop": "true", "crop_margin": "5%"},

                headers={"X-Api-Key": settings.REMOVE_BG_API_KEY},

                timeout=30

            )

            if response.status_code == 402:

                logger.error("remove.bg credits exhausted")

                raise BackgroundRemovalError(

                    "Background removal service is unavailable. Please try again later."

                )

            if response.status_code == 429:

                logger.warning("remove.bg rate limit hit")

                raise BackgroundRemovalError(

                    "Too many requests. Please wait a moment and try again."

                )

            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")

            if "image" not in content_type:

                logger.error(f"remove.bg returned non-image content: {content_type}")

                raise BackgroundRemovalError()

            elapsed = time.perf_counter() - start

            logger.info(f"Background removed in {elapsed:.2f}s")

            return response.content

        except BackgroundRemovalError:

            raise  # don't retry on 402/429/bad content — those won't fix themselves

        except requests.Timeout:

            last_exc = BackgroundRemovalError("Background removal timed out. Please try again.")

            logger.warning(f"remove.bg timed out (attempt {attempt})")

        except requests.RequestException as e:

            last_exc = BackgroundRemovalError()

            logger.warning(f"remove.bg request error (attempt {attempt}): {e}")

        except Exception as e:

            logger.error(f"Background removal unexpected error: {e}")

            raise BackgroundRemovalError()

        if attempt <= MAX_RETRIES:

            time.sleep(RETRY_BACKOFF * attempt)

    logger.error("remove.bg failed after all retries")

    raise last_exc