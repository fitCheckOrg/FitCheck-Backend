"""
Builds the GarmentAsset wire representation. Applies the CONFIRMED
anatomical normalization (small-x -> anatomical_right, large-x ->
anatomical_left) — validated 7/7 across the V1 regression set, not
assumed. Persists the silhouette mask to S3. Reuses the existing
image_hash (MD5, computed at wardrobe processing time) as
asset_revision rather than computing a new hash.
"""

import io
import requests
import numpy as np
from PIL import Image

from core.storage.s3 import storage, S3Folders
from core.storage.supabase_client import supabase
from shared.exceptions import ClothingItemNotFoundError
from shared.models.avatar_asset import NormalizedPoint
from shared.models.garment_asset import (
    GarmentAsset, GarmentAssetTexture, GarmentAssetSilhouette,
    GarmentAssetLandmarks, GarmentAssetCapabilities,
)
from tryon.garment_shoulder_region_extraction import get_binary_mask
from tryon.correspondence_shoulders_test import find_garment_underarms
from tryon.piecewise_continuous_test import get_garment_shoulder_points


def _mask_to_png_bytes(mask: np.ndarray) -> bytes:
    img = Image.fromarray((mask.astype(np.uint8)) * 255, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.read()


def _normalize(pt, width, height) -> NormalizedPoint:
    x, y = pt
    return NormalizedPoint(
        x=x / (width - 1) if width > 1 else 0.0,
        y=y / (height - 1) if height > 1 else 0.0,
    )


def build_garment_asset(item_id: str, user_id: str) -> GarmentAsset:
    result = supabase.table("closet_items")\
        .select("item_id, user_id, category, clean_image_url, image_width, image_height, image_hash")\
        .eq("item_id", item_id)\
        .eq("user_id", user_id)\
        .maybe_single()\
        .execute()

    if not result or not result.data:
        raise ClothingItemNotFoundError()

    row = result.data
    garment_bytes = requests.get(row["clean_image_url"]).content
    mask, image = get_binary_mask(garment_bytes)
    w, h = image.size

    left_ua, right_ua = find_garment_underarms(mask, w)
    left_sh, right_sh = get_garment_shoulder_points(mask, image, left_ua, right_ua, w)

    # CONFIRMED normalization rule (7/7 regression-validated):
    # image-space small-x -> anatomical RIGHT, large-x -> anatomical LEFT
    anatomical_right_ua = left_ua if left_ua[0] < right_ua[0] else right_ua
    anatomical_left_ua = right_ua if left_ua[0] < right_ua[0] else left_ua

    shoulders_available = left_sh is not None and right_sh is not None
    anatomical_left_sh = anatomical_right_sh = None
    if shoulders_available:
        anatomical_right_sh = left_sh if left_sh[0] < right_sh[0] else right_sh
        anatomical_left_sh = right_sh if left_sh[0] < right_sh[0] else left_sh

    landmarks = GarmentAssetLandmarks(
        left_underarm=_normalize(anatomical_left_ua, w, h),
        right_underarm=_normalize(anatomical_right_ua, w, h),
        left_shoulder=_normalize(anatomical_left_sh, w, h) if shoulders_available else None,
        right_shoulder=_normalize(anatomical_right_sh, w, h) if shoulders_available else None,
    )

    silhouette_bytes = _mask_to_png_bytes(mask)
    upload = storage.upload(
        silhouette_bytes,
        folder=f"{S3Folders.GARMENT_ASSETS}/{user_id}/{item_id}/silhouette",
        extension="png",
        content_type="image/png",
    )

    return GarmentAsset(
        asset_id=str(row["item_id"]),
        asset_revision=f"md5:{row['image_hash']}",
        category=row["category"],
        texture=GarmentAssetTexture(url=row["clean_image_url"], width=row["image_width"], height=row["image_height"]),
        silhouette=GarmentAssetSilhouette(url=upload["url"]),
        landmarks=landmarks,
        capabilities=GarmentAssetCapabilities(shoulders=shoulders_available),
    )