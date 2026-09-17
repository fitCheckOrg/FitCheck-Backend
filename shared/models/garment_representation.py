from pydantic import BaseModel, Field


class Point(BaseModel):
    x_percent: float = Field(ge=0, le=100)
    y_percent: float = Field(ge=0, le=100)


class GarmentSilhouette(BaseModel):
    """
    The validated primitive this representation is built around.
    Everything else (torso bounds, sleeve bounds) gets DERIVED from
    this + the landmarks at compositing time — never stored as its
    own rectangular region. This is the specific fix for the
    original rectangular-region trap the whole investigation was
    trying to escape.
    """
    contour: list[Point]  # the traced outer boundary


class GarmentLandmarks(BaseModel):
    left_underarm: Point
    right_underarm: Point
    collar_top: Point
    hem_bottom: Point


class GarmentRepresentationV3(BaseModel):
    version: int = 3
    garment_type: str  # GPT-provided, descriptive only

    silhouette: GarmentSilhouette
    landmarks: GarmentLandmarks

    # Deliberately NO torso_region / left_sleeve_region / 
    # right_sleeve_region fields here. Those get computed from 
    # silhouette + landmarks by the compositor, not persisted — 
    # this is the architectural correction from the previous 
    # message, not an oversight.