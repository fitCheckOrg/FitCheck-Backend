from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    def __init__(self):
        # AI
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
      
        # Supabase
        self.SUPABASE_URL = os.getenv("SUPABASE_URL")
        self.SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  
        
        # AWS
        self.AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
        self.AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
        self.AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")
        
        # External APIs
        self.REMOVE_BG_API_KEY = os.getenv("REMOVE_BG_API_KEY")
        
        self.OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
        self.OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
        self.WEATHER_TIMEOUT = int(os.getenv("WEATHER_TIMEOUT", 5))

        self.STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
        self.STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
        self.STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID")
        self.APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8000")

        self.DEV_AUTH_BYPASS = os.getenv("DEV_AUTH_BYPASS", "false").lower() == "true"
        
        # Storage folders
        self.S3_AVATAR_FOLDER = os.getenv("S3_AVATAR_FOLDER", "avatars")
        self.S3_WARDROBE_FOLDER = os.getenv("S3_WARDROBE_FOLDER", "wardrobe")
        
        # App
        self.PORT = int(os.getenv("PORT", 8000))
        self.ENV = os.getenv("ENV", "development")
        self.MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", 10))
        
        # Image validation
        self.MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", 10))
        self.MIN_IMAGE_DIMENSION = int(os.getenv("MIN_IMAGE_DIMENSION", 300))
        self.MAX_IMAGE_DIMENSION = int(os.getenv("MAX_IMAGE_DIMENSION", 6000))
        self.BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", 5.0))
        
        # Image enhancement
        self.CONTRAST_FACTOR = float(os.getenv("CONTRAST_FACTOR", 1.1))
        self.BRIGHTNESS_FACTOR = float(os.getenv("BRIGHTNESS_FACTOR", 1.05))
        self.COLOR_FACTOR = float(os.getenv("COLOR_FACTOR", 1.1))
        self.SHARPNESS_FACTOR = float(os.getenv("SHARPNESS_FACTOR", 1.3))
        self.MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", 1600))
        
        #freemieum
        self.FREE_TIER_LIMIT = int(os.getenv("FREE_TIER_LIMIT", 30))
        
        #avater
        self.MIN_BODY_CONFIDENCE = float(os.getenv("MIN_BODY_CONFIDENCE", 0.4))
        
        # Outfit compatibility weights
        self.COLOR_WEIGHT = float(os.getenv("COLOR_WEIGHT", 0.40))
        self.STYLE_WEIGHT = float(os.getenv("STYLE_WEIGHT", 0.30))
        self.OCCASION_WEIGHT = float(os.getenv("OCCASION_WEIGHT", 0.15))
        self.SEASON_WEIGHT = float(os.getenv("SEASON_WEIGHT", 0.15))
        
        self.RECENTLY_WORN_DAYS = int(os.getenv("RECENTLY_WORN_DAYS", 3))
        self.ROTATION_PENALTY = float(os.getenv("ROTATION_PENALTY", 8))
        self.FAVORITE_BONUS = float(os.getenv("FAVORITE_BONUS", 3))
        self.TIMES_WORN_WEIGHT = float(os.getenv("TIMES_WORN_WEIGHT", 1.5))
        self.ROTATION_SCORE_BAND = float(os.getenv("ROTATION_SCORE_BAND", 3))
        
        self.POSE_MODEL_PATH = os.getenv("POSE_MODEL_PATH", "models/pose_landmarker.task")
        required = {
        "OPENAI_API_KEY": self.OPENAI_API_KEY,
        "AWS_BUCKET_NAME": self.AWS_BUCKET_NAME,
        "AWS_ACCESS_KEY_ID": self.AWS_ACCESS_KEY_ID,
        "AWS_SECRET_ACCESS_KEY": self.AWS_SECRET_ACCESS_KEY,
        "REMOVE_BG_API_KEY": self.REMOVE_BG_API_KEY,
        "SUPABASE_URL": self.SUPABASE_URL,
        "SUPABASE_SERVICE_KEY": self.SUPABASE_SERVICE_KEY,
        }
        
        missing =  [k for k, v in required.items() if not v]
        if missing:
            raise RuntimeError(
                f"Missing environment variables: {', '.join(missing)}"
            )
    
settings = Settings()