class FitCheckException(Exception):
    """Base exception for all FitCheck errors"""
    def __init__(
        self,
        message: str,
        code: str = "FITCHECK_ERROR",
        status_code: int = 500
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        super().__init__(message)


# ── Image Validation ──────────────────────────────
class ImageTooLargeError(FitCheckException):
    """Image exceeds maximum allowed size"""
    def __init__(self, message="Image too large. Maximum size is 10MB."):
        super().__init__(message, code="IMAGE_TOO_LARGE", status_code=400)

class InvalidImageFormatError(FitCheckException):
    """Image format not supported"""
    def __init__(self, message="Invalid format. Please upload a JPEG, PNG or WEBP."):
        super().__init__(message, code="INVALID_IMAGE_FORMAT", status_code=400)

class BlurryImageError(FitCheckException):
    """Image is too blurry to process"""
    def __init__(self, message="Image is too blurry. Please take a clearer photo."):
        super().__init__(message, code="BLURRY_IMAGE", status_code=400)


# ── AI ────────────────────────────────────────────
class AIAnalysisError(FitCheckException):
    """GPT-4o analysis failed"""
    def __init__(self, message="AI analysis failed. Please try again."):
        super().__init__(message, code="AI_ANALYSIS_ERROR", status_code=500)

class BackgroundRemovalError(FitCheckException):
    """Background removal failed"""
    def __init__(self, message="Background removal failed. Please try again."):
        super().__init__(message, code="BACKGROUND_REMOVAL_ERROR", status_code=500)

class OpenAIConnectionError(FitCheckException):
    """Cannot connect to OpenAI"""
    def __init__(self, message="AI service unavailable. Please try again later."):
        super().__init__(message, code="OPENAI_CONNECTION_ERROR", status_code=503)


# ── Storage ───────────────────────────────────────
class StorageError(FitCheckException):
    """S3 upload or delete failed"""
    def __init__(self, message="Storage operation failed. Please try again."):
        super().__init__(message, code="STORAGE_ERROR", status_code=500)


# ── Avatar ────────────────────────────────────────
class AvatarNotFoundError(FitCheckException):
    """Avatar doesn't exist for this user"""
    def __init__(self, message="Avatar not found. Please set up your avatar first."):
        super().__init__(message, code="AVATAR_NOT_FOUND", status_code=404)


# ── Wardrobe ──────────────────────────────────────
class ClothingItemNotFoundError(FitCheckException):
    """Closet item doesn't exist"""
    def __init__(self, message="Clothing item not found."):
        super().__init__(message, code="CLOTHING_ITEM_NOT_FOUND", status_code=404)

class FreeTierLimitError(FitCheckException):
    """Free user hit 30 item limit"""
    def __init__(self, message="You've reached the free tier limit of 30 items. Upgrade to Premium for unlimited storage."):
        super().__init__(message, code="FREE_TIER_LIMIT", status_code=403)

class DuplicateClothingItemError(FitCheckException):
    """Same item already exists in wardrobe"""
    def __init__(self, message="This item already exists in your wardrobe."):
        super().__init__(message, code="DUPLICATE_ITEM", status_code=409)


# ── Subscription ──────────────────────────────────
class PremiumFeatureError(FitCheckException):
    """Feature requires premium subscription"""
    def __init__(self, message="This feature requires a FitCheck Premium subscription."):
        super().__init__(message, code="PREMIUM_REQUIRED", status_code=403)

class SubscriptionExpiredError(FitCheckException):
    """Subscription has expired"""
    def __init__(self, message="Your subscription has expired. Please renew to continue."):
        super().__init__(message, code="SUBSCRIPTION_EXPIRED", status_code=403)


# ── Auth ──────────────────────────────────────────
class AuthenticationError(FitCheckException):
    """Authentication failed"""
    def __init__(self, message="Authentication failed. Please log in again."):
        super().__init__(message, code="AUTHENTICATION_ERROR", status_code=401)

class AuthorizationError(FitCheckException):
    """User not authorized for this action"""
    def __init__(self, message="You are not authorized to perform this action."):
        super().__init__(message, code="AUTHORIZATION_ERROR", status_code=403)


# ── Rate Limiting ─────────────────────────────────
class RateLimitExceededError(FitCheckException):
    """Too many requests"""
    def __init__(self, message="Too many requests. Please wait a moment and try again."):
        super().__init__(message, code="RATE_LIMIT_EXCEEDED", status_code=429)
        
class GarmentUnsupportedError(FitCheckException):
    """Garment representation built successfully, but shoulders_available
    is False — expected V1 capability gap, not a server error."""
    def __init__(self, message="This garment is not supported by the current try-on fitting pipeline."):
        super().__init__(message, code="GARMENT_UNSUPPORTED", status_code=409)