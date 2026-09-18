"""
Builds the AvatarAsset wire representation from an already-constructed
AvatarRepresentation. Persists torso/upper_arms masks to S3. Normalizes
geometry using x/(width-1), y/(height-1) per the locked contract.

Does not implement /api/avatar/me. Separate representation, separate
endpoint.
"""

import io
import numpy as np
from PIL import Image

from core.storage.s3 import storage, S3Folders
from shared.models.avatar_asset import (
    AvatarAsset, AvatarAssetImage, AvatarAssetLandmarks,
    AvatarAssetBodyRegions, BodyRegion, MaskGeometry, NormalizedPoint,
)
from tryon.avatar_representation import build_avatar_representation


def _mask_to_png_bytes(mask: np.ndarray) -> bytes:
    """Boolean mask -> single-channel PNG bytes."""
    img = Image.fromarray((mask.astype(np.uint8)) * 255, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.read()


def _normalize(pt: tuple[int, int], width: int, height: int) -> NormalizedPoint:
    x, y = pt
    return NormalizedPoint(
        x=x / (width - 1) if width > 1 else 0.0,
        y=y / (height - 1) if height > 1 else 0.0,
    )


def build_avatar_asset(user_id: str) -> AvatarAsset:
    rep = build_avatar_representation(user_id)
    width, height = rep.image.size

    landmarks = AvatarAssetLandmarks(
        left_shoulder=_normalize(rep.left_shoulder, width, height),
        right_shoulder=_normalize(rep.right_shoulder, width, height),
        left_underarm=_normalize(rep.left_underarm, width, height),
        right_underarm=_normalize(rep.right_underarm, width, height),
    )

    torso_bytes = _mask_to_png_bytes(rep.torso_mask)
    upper_arms_bytes = _mask_to_png_bytes(rep.upper_arms_mask)

    torso_upload = storage.upload(
        torso_bytes,
        folder=f"{S3Folders.AVATAR_ASSETS}/{user_id}/torso",
        extension="png",
        content_type="image/png",
    )
    upper_arms_upload = storage.upload(
        upper_arms_bytes,
        folder=f"{S3Folders.AVATAR_ASSETS}/{user_id}/upper_arms",
        extension="png",
        content_type="image/png",
    )

    body_regions = AvatarAssetBodyRegions(
        torso=BodyRegion(geometry=MaskGeometry(url=torso_upload["url"])),
        upper_arms=BodyRegion(geometry=MaskGeometry(url=upper_arms_upload["url"])),
    )

    return AvatarAsset(
        asset_id=str(rep.avatar_id),
        version=rep.avatar_version,
        image=AvatarAssetImage(url=rep.processed_photo_url, width=width, height=height),
        landmarks=landmarks,
        body_regions=body_regions,
    )