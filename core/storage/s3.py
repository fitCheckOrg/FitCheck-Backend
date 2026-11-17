"""
S3 storage.

Try-On v2 T2-1 B5 (2026-11-18) - private images:
    PRIVATE  tryons/...                      try-on renders
             {avatars}/{user}/original/...   avatar photos
             {avatars}/{user}/processed/...
             {wardrobe}/{user}/original/...  garment originals
    PUBLIC   everything else - clean cut-outs ({wardrobe}/{user}/clean/...) and
             profile photos stay public on purpose (clothes only / chosen to be public).

    * The database keeps storing the plain URL / key (public_url) - it's an
      identifier, never shown as-is for a private image.
    * get_url(key) -> a presigned link (1 hour) for a private key, the plain
      URL for a public one. Signed links are reused while they have > 15 min left,
      so the same image keeps the same link (the app's image cache stays warm).
    * url_for(url_or_key) does the same for a stored URL (e.g. avatars.original_photo_url).
    * signed(key) -> {key, url, expires_at} for image objects / POST /media/sign.
Signing is local (no network call). The IAM user needs s3:GetObject.
"""

import logging
import re
import threading
import uuid
from datetime import datetime, timedelta, timezone

import boto3
from botocore.config import Config

from core.config.settings import settings

logger = logging.getLogger(__name__)

SIGNED_SECONDS = 3600              # a signed link works for 1 hour
REUSE_MIN_LEFT = 900               # reuse a cached link while it has >= 15 min left


class S3Folders:
    AVATARS = settings.S3_AVATAR_FOLDER
    WARDROBE = settings.S3_WARDROBE_FOLDER
    OUTFITS = "outfits"
    TRYON = "tryons"
    AVATAR_ASSETS = "avatar-assets"
    GARMENT_ASSETS = "garment-assets"
    TEMP = "temp"


_PRIVATE = re.compile(
    r"^(?:"
    + re.escape(S3Folders.TRYON) + r"/"
    + r"|" + re.escape(S3Folders.AVATARS) + r"/[^/]+/(?:original|processed)/"
    + r"|" + re.escape(S3Folders.WARDROBE) + r"/[^/]+/original/"
    + r")"
)


class S3Storage:
    def __init__(self):
        self.region = settings.AWS_REGION
        self.bucket = settings.AWS_BUCKET_NAME
        self.client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=self.region,
            endpoint_url=f"https://s3.{self.region}.amazonaws.com",
            config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
        )
        self.base = f"https://{self.bucket}.s3.{self.region}.amazonaws.com/"
        self._signed: dict[str, tuple[str, datetime]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------ write / delete

    def upload(
        self,
        image_bytes: bytes,
        folder: str,
        extension: str = "png",
        content_type: str = "image/png"
    ) -> dict:
        """Upload image to S3 - returns key and the plain URL (stored in the database)."""
        try:
            file_key = f"{folder}/{uuid.uuid4()}.{extension}"

            self.client.put_object(
                Bucket=self.bucket,
                Key=file_key,
                Body=image_bytes,
                ContentType=content_type
            )

            return {"key": file_key, "url": self.public_url(file_key)}

        except Exception as e:
            logger.exception(f"S3 upload failed: {e}")
            raise

    def delete(self, file_key: str) -> bool:
        """Delete file from S3 by key"""
        try:
            self.client.delete_object(
                Bucket=self.bucket,
                Key=file_key
            )
            with self._lock:
                self._signed.pop(file_key, None)
            return True
        except Exception as e:
            logger.exception(f"S3 delete failed: {e}")
            return False

    # ------------------------------------------------------------ keys and links

    def public_url(self, file_key: str) -> str:
        """The plain (unsigned) URL. Works only for public keys once the bucket is locked."""
        return f"{self.base}{file_key}"

    def key_from_url(self, url_or_key: str | None) -> str | None:
        """Our key from a stored URL (signed or not) or a bare key. None if it isn't ours."""
        if not url_or_key:
            return None
        value = str(url_or_key)
        if value.startswith(self.base):
            return value[len(self.base):].split("?", 1)[0] or None
        if value.startswith("http://") or value.startswith("https://"):
            return None
        return value.split("?", 1)[0].lstrip("/") or None

    @staticmethod
    def is_private(file_key: str | None) -> bool:
        return bool(file_key) and bool(_PRIVATE.match(file_key))

    def presign(self, file_key: str) -> tuple[str, datetime]:
        """(signed url, expires_at). Reused while it has >= REUSE_MIN_LEFT left."""
        now = datetime.now(timezone.utc)
        with self._lock:
            hit = self._signed.get(file_key)
            if hit and (hit[1] - now).total_seconds() >= REUSE_MIN_LEFT:
                return hit
        url = self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": file_key}, ExpiresIn=SIGNED_SECONDS
        )
        entry = (url, now + timedelta(seconds=SIGNED_SECONDS - 5))
        with self._lock:
            if len(self._signed) > 20000:
                self._signed.clear()
            self._signed[file_key] = entry
        return entry

    def signed(self, file_key: str) -> dict:
        """{key, url, expires_at}. expires_at is None for a public key (the link never expires)."""
        if self.is_private(file_key):
            url, expires = self.presign(file_key)
            return {"key": file_key, "url": url, "expires_at": expires.isoformat()}
        return {"key": file_key, "url": self.public_url(file_key), "expires_at": None}

    def get_url(self, file_key: str) -> str:
        """A link that works: signed for a private key, plain for a public one."""
        return self.signed(file_key)["url"]

    def url_for(self, url_or_key: str | None) -> str | None:
        """Same as get_url, for a URL stored in the database. Foreign URLs come back unchanged."""
        key = self.key_from_url(url_or_key)
        if key is None:
            return url_or_key
        return self.get_url(key)


storage = S3Storage()