"""
HOME-1: weather for the Home screen.

GET /api/weather?lat=9.07&lng=7.49
  200 {"success": true, "data": {temp_celsius, feels_like, condition, icon_key,
                                 is_day, description, humidity, wind_speed, retrieved_at}}
  503 WEATHER_UNAVAILABLE (+ Retry-After: 60)  -> hide the weather widget
condition: clear | cloudy | rain | snow | fog
icon_key:  "<condition>_day" | "<condition>_night"

Results are cached for 10 minutes per ~1 km area (lat/lng rounded to 2 dp)
to stay inside the OpenWeatherMap free quota.
"""

import time
from uuid import UUID

from fastapi import APIRouter, Depends, Query

from core.weather.client import weather_client
from shared.dependencies import get_current_user_id
from shared.exceptions import FitCheckException
from shared.logger import get_logger
from shared.models.responses import SuccessResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/weather", tags=["Weather"])

CACHE_TTL_S = 600
CACHE_MAX = 1000
_cache: dict[tuple[float, float], tuple[float, dict]] = {}

PUBLIC_FIELDS = (
    "temp_celsius", "feels_like", "condition", "icon_key", "is_day",
    "description", "humidity", "wind_speed", "retrieved_at",
)


@router.get("", response_model=SuccessResponse)
async def get_weather(
    lat: float = Query(..., ge=-90, le=90),
    lng: float = Query(..., ge=-180, le=180),
    user_id: UUID = Depends(get_current_user_id),
):
    area = (round(lat, 2), round(lng, 2))
    now = time.time()

    hit = _cache.get(area)
    if hit and now - hit[0] < CACHE_TTL_S:
        data = hit[1]
    else:
        data = await weather_client.get_weather(lat, lng)
        if data is None:
            raise FitCheckException(
                "Weather is unavailable right now.",
                code="WEATHER_UNAVAILABLE",
                status_code=503,
                headers={"Retry-After": "60"},
            )
        if len(_cache) >= CACHE_MAX:
            _cache.clear()
        _cache[area] = (now, data)

    return SuccessResponse(data={k: data.get(k) for k in PUBLIC_FIELDS})