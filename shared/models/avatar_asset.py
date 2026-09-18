"""
AvatarAsset — locked wire contract (Contract A, v1).

Reusable, device-consumable avatar representation for the local 2.5D
Try-On engine. Backend-internal model outputs (SCHP, MediaPipe) do not
appear here — only semantic geometry.
"""

from pydantic import BaseModel, Field


class NormalizedPoint(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class AssetDimensions(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class AvatarAssetImage(BaseModel):
    url: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class AvatarAssetLandmarks(BaseModel):
    left_shoulder: NormalizedPoint
    right_shoulder: NormalizedPoint
    left_underarm: NormalizedPoint
    right_underarm: NormalizedPoint


class MaskGeometry(BaseModel):
    type: str = "mask"
    url: str


class BodyRegion(BaseModel):
    geometry: MaskGeometry


class AvatarAssetBodyRegions(BaseModel):
    torso: BodyRegion
    upper_arms: BodyRegion


class AvatarAsset(BaseModel):
    schema_version: int = 1
    asset_id: str
    version: int
    image: AvatarAssetImage
    landmarks: AvatarAssetLandmarks
    body_regions: AvatarAssetBodyRegions