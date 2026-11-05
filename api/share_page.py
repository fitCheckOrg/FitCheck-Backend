"""
Share link pages (E20, 2026-11-04). PUBLIC - no login. Work lives in
pipelines/share_page_pipeline.py. Included in main.py WITHOUT the /api prefix.

GET /p/{post_id}         HTML page
GET /p/{post_id}/og.png  Open Graph image (collage only)
GET /u/{handle}          public profile page
GET /.well-known/assetlinks.json
GET /.well-known/apple-app-site-association
"""

from uuid import UUID

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse, Response

from pipelines.share_page_pipeline import (
    post_page, profile_page, og_image, not_found_page, assetlinks, apple_app_site_association,
)

router = APIRouter(tags=["Share pages"])


@router.get("/p/{post_id}", response_class=HTMLResponse, include_in_schema=False)
def share_post_page(post_id: UUID):
    page = post_page(str(post_id))
    if not page:
        return HTMLResponse(not_found_page(), status_code=404)
    return HTMLResponse(page, headers={"Cache-Control": "public, max-age=300"})


@router.get("/p/{post_id}/og.png", include_in_schema=False)
def share_post_image(post_id: UUID):
    data = og_image(str(post_id))
    if not data:
        return Response(status_code=404)
    return Response(data, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})


@router.get("/u/{handle}", response_class=HTMLResponse, include_in_schema=False)
def share_profile_page(handle: str):
    page = profile_page(handle)
    if not page:
        return HTMLResponse(not_found_page(), status_code=404)
    return HTMLResponse(page, headers={"Cache-Control": "public, max-age=300"})


@router.get("/.well-known/assetlinks.json", include_in_schema=False)
def well_known_assetlinks():
    data = assetlinks()
    return JSONResponse(data) if data else Response(status_code=404)


@router.get("/.well-known/apple-app-site-association", include_in_schema=False)
def well_known_aasa():
    data = apple_app_site_association()
    return JSONResponse(data) if data else Response(status_code=404)