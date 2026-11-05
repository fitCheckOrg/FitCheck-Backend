from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from core.storage.db_retry import install_db_retry, DB_CONNECTION_ERRORS
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
from api.weather import router as weather_router
from api.collections import router as collections_router
from api.users import router as users_router
from api.comments import router as comments_router, post_router as post_comments_router
from api.activity import router as activity_router
from api.related import router as related_router
from api.search import router as search_router
from api.suggested import router as suggested_router
from api.recreate import router as recreate_router
from api.insights import router as insights_router
from api.share_page import router as share_page_router


# Retry a Supabase query once when the connection drops (see core/storage/db_retry.py).
install_db_retry()

app = FastAPI(title="FitCheck AI Service")

# Global exception handler - no try/except in routes
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
        },
        headers=getattr(exc, "headers", None)
    )

# A dropped Supabase connection that survived the retry -> 503, not 500.
async def db_unavailable_handler(request, exc):
    return JSONResponse(
        status_code=503,
        content={
            "success": False,
            "error": {
                "code": "DB_UNAVAILABLE",
                "message": "We couldn't reach the server just now. Please try again."
            }
        },
        headers={"Retry-After": "5"}
    )

for _db_error in DB_CONNECTION_ERRORS:
    app.add_exception_handler(_db_error, db_unavailable_handler)

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
# E19: insights BEFORE social, so POST /social/posts/views isn't taken by /social/posts/{post_id}
app.include_router(insights_router, prefix="/api")
app.include_router(social_router, prefix="/api")
app.include_router(subscription_router, prefix="/api")
app.include_router(config_router, prefix="/api")
app.include_router(tryon_router, prefix="/api")
app.include_router(weather_router, prefix="/api")
app.include_router(collections_router, prefix="/api")
app.include_router(users_router, prefix="/api")
app.include_router(comments_router, prefix="/api")
app.include_router(post_comments_router, prefix="/api")
app.include_router(activity_router, prefix="/api")
app.include_router(related_router, prefix="/api")
app.include_router(search_router, prefix="/api")
app.include_router(suggested_router, prefix="/api")
app.include_router(recreate_router, prefix="/api")
# E20: public share pages + app-link files, NO /api prefix (fitcheck.app/p/{id})
app.include_router(share_page_router)



@app.get("/")
def health_check():
    return { "status": "FitCheck AI service is running" }

# uvicorn main:app --reload --port 8000