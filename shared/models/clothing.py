from pydantic import BaseModel, UUID4, Field
from typing import Optional
from datetime import datetime, UTC
import uuid
from shared.models.enums import (
    Category, Pattern, Style, Season, Occasion
)
class ClothingItem(BaseModel):
    item_id: UUID4 = Field(default_factory=uuid.uuid4)
    user_id: UUID4
    original_image_url: str
    clean_image_url: str
    category: Category
    item_type: str
    colors: list[str]
    dominant_color: str
    dominant_hex: str | None = None
    secondary_color: str | None = None
    secondary_hex: str | None = None
    accent_color: str | None = None
    accent_hex: str | None = None
    pattern: Pattern
    style: Style
    seasons: list[Season]
    occasions: list[Occasion]
    brand: str | None = None
    pairs_well_with: list[str]
    material: str | None = None
    image_width: int
    image_height: int
    classification_confidence: float = Field(ge=0, le=1)
    favorite: bool = False
    times_worn: int = 0
    last_worn: datetime | None = None
    is_archived: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    asset_revision: str | None = None

    class Config:
        json_encoders = { uuid.UUID: str }
        
class ClothingClassification(BaseModel):
    category: Category
    item_type: str
    colors: list[str]
    dominant_color: str
    dominant_hex: str | None = None
    secondary_color: str | None = None
    secondary_hex: str | None = None
    accent_color: str | None = None
    accent_hex: str | None = None
    pattern: Pattern
    style: Style
    seasons: list[Season]
    occasions: list[Occasion]
    brand: str | None = None
    pairs_well_with: list[str] = []
    material: str | None = None
    classification_confidence: float = Field(ge=0, le=1)
    category: Category
    item_type: str
    colors: list[str]
    dominant_color: str
    secondary_color: str | None = None
    pattern: Pattern
    style: Style
    seasons: list[Season]
    occasions: list[Occasion]
    brand: str | None = None
    pairs_well_with: list[str] = []
    material: str | None = None
    classification_confidence: float = Field(ge=0, le=1)