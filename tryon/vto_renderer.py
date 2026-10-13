"""
Try-on renderer: the ONLY module in the backend that knows which
try-on model/provider we use.

    renderer = get_renderer()
    png_bytes = renderer.render(person_image, garment_image, category="tops")

person_image / garment_image can be PIL Images, raw bytes, or http(s) URLs.
Swapping provider (another fal model, FASHN direct, self-hosted CatVTON)
means adding a new TryOnRenderer subclass and changing get_renderer().
Nothing else in the backend, and nothing in Flutter, changes.

Smoke test (server must be running on :8000):
    python -m tryon.vto_renderer
"""

import base64
import io
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import requests
from PIL import Image

ImageInput = Union[Image.Image, bytes, str]

VALID_CATEGORIES = {"tops", "bottoms", "one-pieces", "auto"}


class TryOnRenderError(Exception):
    """Provider failed, was rejected, or returned no image."""


# ---------------------------------------------------------------- helpers

def _ensure_fal_key():
    """Make sure FAL_KEY is in the environment, falling back to reading .env."""
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


def _to_pil(img: ImageInput) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, (bytes, bytearray)):
        return Image.open(io.BytesIO(img))
    if isinstance(img, str) and img.startswith("http"):
        r = requests.get(img, timeout=60)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content))
    raise TypeError(f"Unsupported image input: {type(img)}")


def _flatten_on_white(img: Image.Image) -> Image.Image:
    """Our garment PNGs are transparent; models want a clean white background."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        return bg
    return img.convert("RGB")


def _data_uri(img: Image.Image, fmt: str) -> str:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.convert("RGB").save(buf, format="JPEG", quality=95)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG")
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


# ---------------------------------------------------------------- interface

class TryOnRenderer(ABC):
    """Person photo + garment photo -> PNG bytes of the person wearing it."""

    #: short id stored with every cached render, so we know what made it
    name: str = "base"

    @abstractmethod
    def render(self, person_image: ImageInput, garment_image: ImageInput,
               category: str = "tops") -> bytes:
        ...


# ---------------------------------------------------------------- fal / FASHN

class FalFashnRenderer(TryOnRenderer):
    """FASHN v1.6 via fal.ai. One sample, fixed seed, quality mode."""

    ENDPOINT = "fal-ai/fashn/tryon/v1.6"
    name = "fal-fashn-v1.6"

    def __init__(self, seed: int = 42, mode: str = "quality"):
        self.seed = seed
        self.mode = mode

    def render(self, person_image: ImageInput, garment_image: ImageInput,
               category: str = "tops") -> bytes:
        if category not in VALID_CATEGORIES:
            raise ValueError(f"category must be one of {sorted(VALID_CATEGORIES)}, got {category!r}")
        _ensure_fal_key()
        import fal_client  # imported after FAL_KEY is guaranteed

        person = _to_pil(person_image).convert("RGB")
        garment = _flatten_on_white(_to_pil(garment_image))

        try:
            result = fal_client.subscribe(
                self.ENDPOINT,
                arguments={
                    "model_image": _data_uri(person, "JPEG"),
                    "garment_image": _data_uri(garment, "PNG"),
                    "category": category,
                    "garment_photo_type": "flat-lay",
                    "mode": self.mode,
                    "num_samples": 1,
                    "seed": self.seed,
                    "output_format": "png",
                },
            )
        except Exception as exc:  # fal raises its own HTTP error types
            raise TryOnRenderError(f"{self.name} request failed: {exc}") from exc

        images = (result or {}).get("images") or []
        if not images or not images[0].get("url"):
            raise TryOnRenderError(f"{self.name} returned no image: {result}")

        r = requests.get(images[0]["url"], timeout=60)
        if r.status_code != 200:
            raise TryOnRenderError(f"could not download result ({r.status_code})")
        return r.content


# ---------------------------------------------------------------- factory

def get_renderer() -> TryOnRenderer:
    """The one place that decides which provider the backend uses."""
    return FalFashnRenderer()


# ---------------------------------------------------------------- smoke test

if __name__ == "__main__":
    import time

    USER = "f4b5f129-db34-40e5-8423-d1b256a15578"
    GARMENT = "a852b031-a407-49db-9df6-7e22ecac2ea6"
    API = "http://127.0.0.1:8000/api"

    def _data(url):
        body = requests.get(url, timeout=60).json()
        return body.get("data", body)

    avatar = _data(f"{API}/avatar/me?user_id={USER}")
    garment = _data(f"{API}/wardrobe/{GARMENT}?user_id={USER}")

    renderer = get_renderer()
    t0 = time.time()
    png = renderer.render(avatar["original_photo_url"], garment["clean_image_url"], "tops")
    out = Path("outputs/vto_spike/renderer_smoke.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png)
    print(f"OK  {renderer.name}  {len(png)//1024} KB  {time.time()-t0:.1f}s  -> {out}")