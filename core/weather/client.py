import httpx
from datetime import datetime, UTC
from core.config.settings import settings
from shared.logger import get_logger

logger = get_logger(__name__)

CONDITION_MAP = {
    "rain": "rain",
    "drizzle": "rain",
    "thunderstorm": "rain",
    "snow": "snow",
    "mist": "fog",
    "fog": "fog",
    "haze": "fog",
    "smoke": "fog",
    "dust": "fog",
    "sand": "fog",
    "clear": "clear",
    "clouds": "cloudy",
}


class WeatherClient:
    """
    Thin wrapper around OpenWeatherMap.
    No fashion logic here - just fetches and normalizes weather data.
    Never raises - weather is optional context, not critical path.
    """

    def __init__(self):
        self.api_key = settings.OPENWEATHER_API_KEY
        self.base_url = settings.OPENWEATHER_BASE_URL
        self.timeout = settings.WEATHER_TIMEOUT

    async def get_weather(
        self,
        latitude: float | None,
        longitude: float | None
    ) -> dict | None:
        if latitude is None or longitude is None:
            return None

        if not self.api_key:
            logger.warning("OpenWeatherMap API key not configured - skipping weather")
            return None

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    self.base_url,
                    params={
                        "lat": latitude,
                        "lon": longitude,
                        "appid": self.api_key,
                        "units": "metric"
                    }
                )
                response.raise_for_status()
                data = response.json()

            return self._normalize(data)

        except Exception as e:
            logger.warning(f"Weather fetch failed - continuing without it: {e}")
            return None

    def _normalize(self, data: dict) -> dict:
        main = data.get("main", {})
        weather_list = data.get("weather") or [{}]
        first = weather_list[0] if weather_list else {}
        raw_condition = (first.get("main") or "").lower()
        condition = CONDITION_MAP.get(raw_condition, "cloudy")
        wind = data.get("wind", {})

        # OpenWeatherMap icon codes end in "d" (day) or "n" (night), e.g. "10n".
        icon = first.get("icon") or ""
        is_day = not icon.endswith("n")

        temp = main.get("temp")

        return {
            "temp_celsius": temp,
            "feels_like": main.get("feels_like"),
            "condition": condition,
            "icon_key": f"{condition}_{'day' if is_day else 'night'}",
            "is_day": is_day,
            "description": first.get("description"),
            "is_raining": condition == "rain",
            "is_snowing": condition == "snow",
            "is_windy": (wind.get("speed") or 0) > 8,
            "humidity": main.get("humidity"),
            "wind_speed": wind.get("speed"),
            "is_hot": temp is not None and temp >= 28,
            "is_cold": temp is not None and temp <= 12,
            "is_mild": temp is not None and 12 < temp < 28,
            "retrieved_at": datetime.now(UTC).isoformat(),
        }


weather_client = WeatherClient()