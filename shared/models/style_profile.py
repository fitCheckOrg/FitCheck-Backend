from pydantic import BaseModel, Field
from datetime import datetime, UTC
from uuid import UUID


class StyleProfile(BaseModel):
    # Optional: the server always uses the user from the auth token.
    user_id: UUID | None = None
    styles: list[str] = []
    preferred_colors: list[str] = []
    disliked_colors: list[str] = []
    favorite_occasions: list[str] = []
    self_reported_body_shape: str | None = None
    onboarding_completed: bool = False


class UpdateStyleProfileRequest(BaseModel):
    styles: list[str] | None = None
    preferred_colors: list[str] | None = None
    disliked_colors: list[str] | None = None
    favorite_occasions: list[str] | None = None
    self_reported_body_shape: str | None = None
    onboarding_completed: bool | None = None