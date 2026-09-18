from fastapi import APIRouter, UploadFile, File, Form, Query, Depends
from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from datetime import datetime, UTC
from shared.logger import get_logger
from shared.models.responses import SuccessResponse
from shared.models.enums import Category
from shared.exceptions import ClothingItemNotFoundError
from core.storage.supabase_client import supabase
from core.storage.s3 import storage
from pipelines.wardrobe_pipeline import run_wardrobe_pipeline
from shared.dependencies import get_current_user_id
from shared.models.clothing import ClothingItem
from core.constants.wardrobe_select import WARDROBE_ITEM_SELECT

from shared.models.garment_asset import GarmentAsset
from workers.wardrobe.asset_builder import build_garment_asset

logger = get_logger(__name__)

router = APIRouter(prefix="/wardrobe", tags=["Wardrobe"])


class UpdateClothingItemRequest(BaseModel):
    brand: str | None = None
    material: str | None = None
    item_type: str | None = None


# ── Fixed routes first (before dynamic /{item_id}) ────────────────

@router.get("/health")
async def health():
    return { "service": "wardrobe", "version": "1.0.0", "status": "healthy" }


@router.get("/stats/summary", response_model=SuccessResponse)
async def get_wardrobe_stats(user_id: UUID = Depends(get_current_user_id)):
    """Get closet stats — count by category"""
    logger.info(f"Get stats user={user_id}")

    result = supabase.table("closet_items")\
        .select("category")\
        .eq("user_id", str(user_id))\
        .eq("is_archived", False)\
        .execute()

    counts = {}
    for item in result.data:
        cat = item["category"]
        counts[cat] = counts.get(cat, 0) + 1

    return SuccessResponse(data={
        "total": len(result.data),
        "by_category": counts
    })


# ── Collection routes ─────────────────────────────────────────────

@router.post("/upload", response_model=SuccessResponse)
async def upload_clothing_item(
    file: UploadFile = File(...),
    user_id: UUID = Depends(get_current_user_id)
):
    """Upload and process a clothing item"""
    logger.info(f"Wardrobe upload user={user_id}")
    item = await run_wardrobe_pipeline(file=file, user_id=str(user_id))
    return SuccessResponse(data=item.model_dump())


@router.get("/", response_model=SuccessResponse)
async def get_wardrobe(
    user_id: UUID = Depends(get_current_user_id),
    category: Optional[Category] = Query(None),
    search: Optional[str] = Query(None),
    favorites_only: bool = Query(False),
    include_archived: bool = Query(False),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    sort: str = Query("newest")
):
    """Get all closet items for a user"""
    logger.info(f"Get wardrobe user={user_id} category={category} search={search}")

    query = supabase.table("closet_items")\
    .select(WARDROBE_ITEM_SELECT, count="exact")\
    .eq("user_id", str(user_id))

    if category:
        query = query.eq("category", category.value)
    if favorites_only:
        query = query.eq("favorite", True)
    if not include_archived:
        query = query.eq("is_archived", False)
    if search:
        query = query.ilike("search_text", f"%{search}%")

    sort_map = {
        "newest": ("created_at", True),
        "oldest": ("created_at", False),
        "most_worn": ("times_worn", True),
        "least_worn": ("times_worn", False),
    }
    sort_col, sort_desc = sort_map.get(sort, ("created_at", True))
    query = query.order(sort_col, desc=sort_desc)
    query = query.range(offset, offset + limit - 1)

    result = query.execute()
    total = result.count or 0
    items = [ClothingItem(**row).model_dump(mode="json") for row in result.data]
    return SuccessResponse(data={
        "items": items,
        "pagination": {
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total
        }
    })


# ── Item routes ──────────

@router.get("/{item_id}", response_model=SuccessResponse)
async def get_wardrobe_item(item_id: UUID, user_id: UUID = Depends(get_current_user_id)):
    result = supabase.table("closet_items")\
        .select(WARDROBE_ITEM_SELECT)\
        .eq("item_id", str(item_id))\
        .eq("user_id", str(user_id))\
        .maybe_single()\
        .execute()

    if not result or not result.data:
        raise ClothingItemNotFoundError()

    item = ClothingItem(**result.data).model_dump(mode="json")
    return SuccessResponse(data=item)

@router.patch("/{item_id}", response_model=SuccessResponse)
async def update_wardrobe_item(
    item_id: UUID,
    body: UpdateClothingItemRequest,
    user_id: UUID = Depends(get_current_user_id)
):
    """Edit clothing item metadata"""
    logger.info(f"Update item={item_id} user={user_id}")

    updates = {}
    if body.brand is not None: updates["brand"] = body.brand
    if body.material is not None: updates["material"] = body.material
    if body.item_type is not None: updates["item_type"] = body.item_type

    if not updates:
        return SuccessResponse(data={"message": "Nothing to update"})

    result = supabase.table("closet_items")\
        .update(updates)\
        .eq("item_id", str(item_id))\
        .eq("user_id", str(user_id))\
        .execute()
    
    if not result.data:
        raise ClothingItemNotFoundError()

    item = ClothingItem(**result.data[0]).model_dump(mode="json")
    return SuccessResponse(data=item)



@router.delete("/{item_id}", response_model=SuccessResponse)
async def delete_wardrobe_item(item_id: UUID, user_id: UUID = Depends(get_current_user_id)):
    """Delete clothing item + S3 images"""
    logger.info(f"Delete item={item_id} user={user_id}")

    # Get item first
    result = supabase.table("closet_items")\
        .select("original_image_url, clean_image_url")\
        .eq("item_id", str(item_id))\
        .eq("user_id", str(user_id))\
        .single()\
        .execute()

    if not result.data:
        raise ClothingItemNotFoundError()

    item = result.data

    # Delete S3 first
    for url_key in ["original_image_url", "clean_image_url"]:
        url = item.get(url_key)
        if url:
            key = url.split(".amazonaws.com/")[-1]
            storage.delete(key)
            logger.info(f"S3 deleted: {key}")

    #delete from DB
    supabase.table("closet_items")\
        .delete()\
        .eq("item_id", str(item_id))\
        .eq("user_id", str(user_id))\
        .execute()

    return SuccessResponse(data={"message": "Item deleted successfully"})


@router.patch("/{item_id}/favorite", response_model=SuccessResponse)
async def toggle_favorite(item_id: UUID, user_id: UUID = Depends(get_current_user_id)):
    result = supabase.table("closet_items")\
        .select("favorite")\
        .eq("item_id", str(item_id))\
        .eq("user_id", str(user_id))\
        .single()\
        .execute()

    if not result.data:
        raise ClothingItemNotFoundError()

    new_status = not result.data["favorite"]

    supabase.table("closet_items")\
        .update({"favorite": new_status})\
        .eq("item_id", str(item_id))\
        .eq("user_id", str(user_id))\
        .execute()

    return SuccessResponse(data={"item_id": str(item_id), "favorite": new_status})

@router.patch("/{item_id}/worn", response_model=SuccessResponse)
async def log_worn(item_id: UUID, user_id: UUID = Depends(get_current_user_id)):
    """Log item as worn today"""
    logger.info(f"Log worn item={item_id} user={user_id}")

    result = supabase.table("closet_items")\
        .select("times_worn")\
        .eq("item_id", str(item_id))\
        .eq("user_id", str(user_id))\
        .single()\
        .execute()

    if not result.data:
        raise ClothingItemNotFoundError()

    current = result.data.get("times_worn") or 0
    new_count = current + 1

    supabase.table("closet_items")\
        .update({
            "times_worn": new_count,
            "last_worn": datetime.now(UTC).isoformat()
        })\
        .eq("item_id", str(item_id))\
        .eq("user_id", str(user_id))\
        .execute()

    return SuccessResponse(data={"item_id": str(item_id), "times_worn": new_count})


@router.patch("/{item_id}/archive", response_model=SuccessResponse)
async def toggle_archive(item_id: UUID, user_id: UUID = Depends(get_current_user_id)):
    """Toggle archive status"""
    logger.info(f"Toggle archive item={item_id} user={user_id}")

    result = supabase.table("closet_items")\
        .select("is_archived")\
        .eq("item_id", str(item_id))\
        .eq("user_id", str(user_id))\
        .single()\
        .execute()

    if not result.data:
        raise ClothingItemNotFoundError()

    new_status = not result.data["is_archived"]

    supabase.table("closet_items")\
        .update({"is_archived": new_status})\
        .eq("item_id", str(item_id))\
        .eq("user_id", str(user_id))\
        .execute()

    return SuccessResponse(data={"item_id": str(item_id), "is_archived": new_status})

@router.get("/{item_id}/asset")
async def get_garment_asset(item_id: UUID, user_id: UUID = Depends(get_current_user_id)):
    logger.info(f"Get garment asset — item={item_id} user={user_id}")
    asset = build_garment_asset(str(item_id), str(user_id))
    return SuccessResponse(data=asset)