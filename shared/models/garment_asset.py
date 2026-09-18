"""
GarmentAsset — locked wire contract (Contract B, v1).

Reusable, avatar-independent garment representation. Landmarks are
anatomically normalized at serialization time (confirmed 7/7 on the
V1 regression set — see verify_garment_orientation_regression.py).
Shoulders are optional; capabilities.shoulders signals fitting
readiness explicitly, per the locked contract.
"""

from typing import Optional
from pydantic import BaseModel

from shared.models.avatar_asset import NormalizedPoint


class GarmentAssetTexture(BaseModel):
    url: str
    width: int
    height: int


class GarmentAssetSilhouette(BaseModel):
    type: str = "mask"
    url: str


class GarmentAssetLandmarks(BaseModel):
    left_underarm: NormalizedPoint
    right_underarm: NormalizedPoint
    left_shoulder: Optional[NormalizedPoint] = None
    right_shoulder: Optional[NormalizedPoint] = None


class GarmentAssetCapabilities(BaseModel):
    shoulders: bool


class GarmentAsset(BaseModel):
    schema_version: int = 1
    asset_id: str
    asset_revision: str
    category: str
    texture: GarmentAssetTexture
    silhouette: GarmentAssetSilhouette
    landmarks: GarmentAssetLandmarks
    capabilities: GarmentAssetCapabilities