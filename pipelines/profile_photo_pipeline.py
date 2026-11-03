"""
Public profile photo (Discover part 2, B10.4 - 2026-11-03).

This is the ONLY image that may ever appear in a user object's avatar_url.
It is a separate upload from the try-on avatar photo and is never linked to it:
own columns (profiles.photo_key, profiles.photo_small_key), own S3 folder
(profile-photos/<user_id>/...), own code path.

Processing (all in memory, nothing of the original is stored):
    * open with Pillow, apply the EXIF rotation, then DROP all metadata
      (EXIF, GPS, camera info) - the saved JPEGs are new images with none
    * crop to a square: the client's crop box if given, else the centre
    * save two sizes: 400x400 (avatar_url) and 96x96 (avatar_url_small)
    * a new upload replaces (and deletes) the previous photo
"""

import io
import logging
from uuid import UUID

from PIL import Image, ImageOps, UnidentifiedImageError

from core.storage.s3 import storage
from core.storage.supabase_client import supabase
from shared.exceptions import FitCheckException

logger = logging.getLogger(__name__)

# HEIC (iPhone) only if pillow-heif is installed; otherwise HEIC -> PHOTO_INVALID.
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORTED = True
except Exception:
    HEIC_SUPPORTED = False

MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 40_000_000            # refuse absurd dimensions (decompression bombs)
MIN_SIDE = 96
SIZES = {"large": 400, "small": 96}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "MPO", "HEIF", "HEIC"}
JPEG_QUALITY = 88
FOLDER = "profile-photos"


def _bad(code: str, message: str, status: int = 400) -> FitCheckException:
    return FitCheckException(message, code=code, status_code=status)


def _photo_invalid(msg: str = "Use a JPG, PNG or WEBP photo.") -> FitCheckException:
    return _bad("PHOTO_INVALID", msg)


def _square_box(w: int, h: int, crop: dict | None) -> tuple[int, int, int, int]:
    """Crop box in the (rotated) image's pixels. Client box is clamped to the image."""
    if crop and all(crop.get(k) is not None for k in ("x", "y", "size")):
        size = max(1, min(int(crop["size"]), w, h))
        x = max(0, min(int(crop["x"]), w - size))
        y = max(0, min(int(crop["y"]), h - size))
        return x, y, x + size, y + size
    side = min(w, h)
    x, y = (w - side) // 2, (h - side) // 2
    return x, y, x + side, y + side


def _process(raw: bytes, crop: dict | None) -> dict[str, bytes]:
    if len(raw) > MAX_BYTES:
        raise _bad("PHOTO_TOO_LARGE", "Photo must be 8 MB or smaller.", 413)
    if not raw:
        raise _photo_invalid("The photo is empty.")
    try:
        img = Image.open(io.BytesIO(raw))
        fmt = (img.format or "").upper()
        if fmt not in ALLOWED_FORMATS:
            raise _photo_invalid()
        if img.width * img.height > MAX_PIXELS:
            raise _photo_invalid("That photo is too big in pixels. Try a smaller one.")
        img = ImageOps.exif_transpose(img)        # respect phone rotation, then forget EXIF
        img.load()
    except FitCheckException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise _photo_invalid("That file isn't a photo we can read. Use JPG, PNG or WEBP.")

    if min(img.width, img.height) < MIN_SIDE:
        raise _photo_invalid(f"Photo must be at least {MIN_SIDE} x {MIN_SIDE} pixels.")

    # Flatten transparency on white, drop every bit of metadata.
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        img = bg
    else:
        img = img.convert("RGB")

    square = img.crop(_square_box(img.width, img.height, crop))
    out = {}
    for name, px in SIZES.items():
        clean = Image.new("RGB", (px, px))                    # brand-new image: no EXIF/GPS
        clean.paste(square.resize((px, px), Image.LANCZOS))
        buf = io.BytesIO()
        clean.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        out[name] = buf.getvalue()
    return out


def _current_keys(uid: str) -> dict:
    res = supabase.table("profiles").select("id, photo_key, photo_small_key")\
        .eq("id", uid).limit(1).execute()
    if not res.data:
        raise _bad("USER_NOT_FOUND", "User not found.", 404)
    return res.data[0]


def _delete_keys(*keys) -> None:
    for k in keys:
        if k:
            storage.delete(k)


def upload_profile_photo(user_id: UUID, raw: bytes, crop: dict | None = None) -> dict:
    uid = str(user_id)
    old = _current_keys(uid)
    images = _process(raw, crop)

    folder = f"{FOLDER}/{uid}"
    large = storage.upload(images["large"], folder=folder, extension="jpg", content_type="image/jpeg")
    small = storage.upload(images["small"], folder=folder, extension="jpg", content_type="image/jpeg")

    try:
        supabase.table("profiles").update({
            "photo_key": large["key"], "photo_small_key": small["key"],
        }).eq("id", uid).execute()
    except Exception:
        _delete_keys(large["key"], small["key"])       # don't leave orphans
        raise

    _delete_keys(old.get("photo_key"), old.get("photo_small_key"))
    logger.info("profile photo set user=%s key=%s", uid, large["key"])
    return {"avatar_url": large["url"], "avatar_url_small": small["url"]}


def delete_profile_photo(user_id: UUID) -> dict:
    """Idempotent: deleting when there's no photo is still 200."""
    uid = str(user_id)
    old = _current_keys(uid)
    supabase.table("profiles").update({"photo_key": None, "photo_small_key": None})\
        .eq("id", uid).execute()
    _delete_keys(old.get("photo_key"), old.get("photo_small_key"))
    return {"avatar_url": None, "avatar_url_small": None}