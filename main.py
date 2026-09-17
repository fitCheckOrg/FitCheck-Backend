from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from api.wardrobe import router as wardrobe_router
from shared.exceptions import FitCheckException
from core.config.settings import settings
from api.avatar import router as avatar_router
from api.style_profile import router as style_profile_router
from api.outfit import router as outfit_router
from api.stylist import router as stylist_router
from api.social import router as social_router
from api.subscription import router as subscription_router
from api.config import router as config_router
from api.tryon import router as tryon_router


app = FastAPI(title="FitCheck AI Service")

# Global exception handler — no try/except in routes
@app.exception_handler(FitCheckException)
async def fitcheck_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.code,
                "message": exc.message
            }
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "Something went wrong. Please try again."
            }
        }
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://10.0.2.2:3000"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(wardrobe_router, prefix="/api")
app.include_router(avatar_router, prefix="/api")
app.include_router(style_profile_router, prefix="/api")
app.include_router(outfit_router, prefix="/api")
app.include_router(stylist_router, prefix="/api")
app.include_router(social_router, prefix="/api")
app.include_router(subscription_router, prefix="/api")
app.include_router(config_router, prefix="/api")
app.include_router(tryon_router, prefix="/api")  # Added tryon router



@app.get("/")
def health_check():
    return { "status": "FitCheck AI service is running" }

# uvicorn main:app --reload --port 8000