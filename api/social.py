from fastapi import APIRouter, Query, Depends
from uuid import UUID
from shared.logger import get_logger
from shared.models.responses import SuccessResponse
from shared.exceptions import FitCheckException
from core.storage.supabase_client import supabase
from pydantic import BaseModel
from typing import Literal
from shared.dependencies import get_current_user_id
from shared.models.enums import DiscoverCategory



logger = get_logger(__name__)

router = APIRouter(prefix="/social", tags=["Social"])
VALID_REACTIONS = {"fire", "clean", "bold"}



class ReactRequest(BaseModel):
    reaction: str

@router.get("/health")
async def health():
    return { "service": "social", "version": "1.1.0", "status": "healthy" }


@router.post("/follow/{target_user_id}", response_model=SuccessResponse)
async def follow_user(
    target_user_id: UUID,
    user_id: UUID = Depends(get_current_user_id)
):
    """Follow another user"""
    if str(target_user_id) == str(user_id):
        raise FitCheckException(
            "You can't follow yourself.",
            code="INVALID_FOLLOW", status_code=400
        )

    logger.info(f"Follow - follower={user_id} following={target_user_id}")

    user_result = supabase.table("profiles")\
        .select("id")\
        .eq("id", str(target_user_id))\
        .single()\
        .execute()

    if not user_result.data:
        raise FitCheckException(
            "User not found.", code="USER_NOT_FOUND", status_code=404
        )

    existing = supabase.table("follows")\
        .select("follower_id")\
        .eq("follower_id", str(user_id))\
        .eq("following_id", str(target_user_id))\
        .execute()

    if existing.data:
        return SuccessResponse(data={
            "message": "Already following",
            "following_id": str(target_user_id)
        })

    result = supabase.table("follows")\
        .insert({
            "follower_id": str(user_id),
            "following_id": str(target_user_id)
        })\
        .execute()

    if not result.data:
        raise FitCheckException(
            "Failed to follow user.", code="DATABASE_ERROR", status_code=500
        )

    counts = _get_follow_counts(str(target_user_id))

    return SuccessResponse(data={
        "message": "Now following",
        "following_id": str(target_user_id),
        **counts
    })


@router.delete("/follow/{target_user_id}", response_model=SuccessResponse)
async def unfollow_user(
    target_user_id: UUID,
    user_id: UUID = Depends(get_current_user_id)
):
    """Unfollow a user"""
    logger.info(f"Unfollow - follower={user_id} following={target_user_id}")

    supabase.table("follows")\
        .delete()\
        .eq("follower_id", str(user_id))\
        .eq("following_id", str(target_user_id))\
        .execute()

    counts = _get_follow_counts(str(target_user_id))

    return SuccessResponse(data={
        "message": "Unfollowed",
        "following_id": str(target_user_id),
        **counts
    })


@router.get("/followers", response_model=SuccessResponse)
async def get_followers(
    user_id: UUID = Depends(get_current_user_id),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Get list of users following this user"""
    result = supabase.table("follows")\
        .select("follower_id, created_at, profiles!follows_follower_id_fkey(id, full_name)", count="exact")\
        .eq("following_id", str(user_id))\
        .order("created_at", desc=True)\
        .range(offset, offset + limit - 1)\
        .execute()

    followers = result.data or []
    total = result.count or 0

    return SuccessResponse(data={
        "followers": followers,
        "pagination": {
            "total": total, "limit": limit, "offset": offset,
            "has_more": (offset + limit) < total
        }
    })


@router.get("/following", response_model=SuccessResponse)
async def get_following(
    user_id: UUID = Depends(get_current_user_id),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Get list of users this user follows"""
    result = supabase.table("follows")\
        .select("following_id, created_at, profiles!follows_following_id_fkey(id, full_name)", count="exact")\
        .eq("follower_id", str(user_id))\
        .order("created_at", desc=True)\
        .range(offset, offset + limit - 1)\
        .execute()

    following = result.data or []
    total = result.count or 0

    return SuccessResponse(data={
        "following": following,
        "pagination": {
            "total": total, "limit": limit, "offset": offset,
            "has_more": (offset + limit) < total
        }
    })


def _get_follow_counts(user_id: str) -> dict:
    """Shared helper - returns current followers/following counts for a user."""
    followers = supabase.table("follows")\
        .select("follower_id", count="exact")\
        .eq("following_id", user_id).execute()

    following = supabase.table("follows")\
        .select("following_id", count="exact")\
        .eq("follower_id", user_id).execute()

    return {
        "followers_count": followers.count or 0,
        "following_count": following.count or 0
    }


@router.post("/reactions/{outfit_id}", response_model=SuccessResponse)
async def react_to_outfit(
    outfit_id: UUID,
    body: ReactRequest,
    user_id: UUID = Depends(get_current_user_id)
):
    """Add or update a reaction on an outfit. One reaction per user per outfit."""
    if body.reaction not in VALID_REACTIONS:
        raise FitCheckException(
            f"Invalid reaction. Must be one of: {', '.join(VALID_REACTIONS)}.",
            code="INVALID_REACTION", status_code=400
        )

    outfit_result = supabase.table("outfits")\
        .select("outfit_id")\
        .eq("outfit_id", str(outfit_id))\
        .single()\
        .execute()

    if not outfit_result.data:
        raise FitCheckException(
            "Outfit not found.", code="OUTFIT_NOT_FOUND", status_code=404
        )

    logger.info(f"React - outfit={outfit_id} user={user_id} reaction={body.reaction}")

    result = supabase.table("outfit_reactions")\
        .upsert({
            "outfit_id": str(outfit_id),
            "user_id": str(user_id),
            "reaction": body.reaction
        }, on_conflict="outfit_id,user_id")\
        .execute()

    if not result.data:
        raise FitCheckException(
            "Failed to save reaction.", code="DATABASE_ERROR", status_code=500
        )

    counts = _get_reaction_counts(str(outfit_id))

    return SuccessResponse(data={
        "outfit_id": str(outfit_id),
        "reaction": body.reaction,
        **counts
    })


@router.delete("/reactions/{outfit_id}", response_model=SuccessResponse)
async def remove_reaction(
    outfit_id: UUID,
    user_id: UUID = Depends(get_current_user_id)
):
    """Remove your reaction from an outfit"""
    logger.info(f"Remove reaction - outfit={outfit_id} user={user_id}")

    supabase.table("outfit_reactions")\
        .delete()\
        .eq("outfit_id", str(outfit_id))\
        .eq("user_id", str(user_id))\
        .execute()

    counts = _get_reaction_counts(str(outfit_id))

    return SuccessResponse(data={
        "outfit_id": str(outfit_id),
        "reaction": None,
        **counts
    })


@router.get("/reactions/{outfit_id}", response_model=SuccessResponse)
async def get_reactions(
    outfit_id: UUID,
    user_id: UUID = Depends(get_current_user_id)
):
    """
    Get reaction counts for an outfit. If user_id is provided,
    also returns that user's current reaction (or null).
    """
    counts = _get_reaction_counts(str(outfit_id))

    my_reaction = None
    if user_id:
        result = supabase.table("outfit_reactions")\
            .select("reaction")\
            .eq("outfit_id", str(outfit_id))\
            .eq("user_id", str(user_id))\
            .execute()
        if result.data:
            my_reaction = result.data[0]["reaction"]

    return SuccessResponse(data={
        "outfit_id": str(outfit_id),
        "my_reaction": my_reaction,
        **counts
    })


def _get_reaction_counts(outfit_id: str) -> dict:
    """Shared helper - returns reaction counts by type for an outfit."""
    counts = {r: 0 for r in VALID_REACTIONS}

    result = supabase.table("outfit_reactions")\
        .select("reaction")\
        .eq("outfit_id", outfit_id)\
        .execute()

    for row in (result.data or []):
        r = row.get("reaction")
        if r in counts:
            counts[r] += 1

    return {"reaction_counts": counts, "total_reactions": sum(counts.values())}


@router.get("/feed", response_model=SuccessResponse)
async def get_discover_feed(
    user_id: UUID = Depends(get_current_user_id),
    sort: Literal["recent", "popular", "following"] = Query("recent"),
    category: DiscoverCategory | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """
    Browse public outfits from the community.
    sort=recent    -> newest public outfits first
    sort=popular    -> most total reactions first
    sort=following  -> public outfits only from users you follow, newest first
    """
    logger.info(f"Get feed - user={user_id} sort={sort} category={category}")

    feed_fields = (
        "outfit_id, user_id, overall_score, overall_grade, "
        "item_ids, occasion, name, created_at, is_public, category"
    )

    if sort == "following":
        follows_result = supabase.table("follows")\
            .select("following_id")\
            .eq("follower_id", str(user_id))\
            .execute()
        following_ids = [f["following_id"] for f in (follows_result.data or [])]

        if not following_ids:
            return SuccessResponse(data={
                "outfits": [],
                "pagination": {"total": 0, "limit": limit, "offset": offset, "has_more": False}
            })

        query = supabase.table("outfits")\
            .select(feed_fields, count="exact")\
            .eq("is_public", True)\
            .in_("user_id", following_ids)\
            .order("created_at", desc=True)\
            .range(offset, offset + limit - 1)

    else:
        query = supabase.table("outfits")\
            .select(feed_fields, count="exact")\
            .eq("is_public", True)\
            .range(offset, offset + limit - 1)

        if sort == "recent":
            query = query.order("created_at", desc=True)

    if category:
        query = query.eq("category", category.value)

    result = query.execute()
    result = query.execute()
    outfits = result.data or []
    total = result.count or 0

    outfit_ids = [o["outfit_id"] for o in outfits]
    creator_ids = list({o["user_id"] for o in outfits})
    all_item_ids = list({item_id for o in outfits for item_id in (o.get("item_ids") or [])})

    reaction_counts_by_outfit = _get_bulk_reaction_counts(outfit_ids)
    creator_profiles = _get_bulk_creator_profiles(creator_ids)
    cover_images_by_outfit_items = _get_bulk_item_images(all_item_ids)
    my_reactions = _get_bulk_my_reactions(outfit_ids, str(user_id))
    my_saved = _get_bulk_saved_status(outfit_ids, str(user_id))
    my_following = _get_bulk_following_status(creator_ids, str(user_id))

    for outfit in outfits:
        oid = outfit["outfit_id"]
        creator_id = outfit["user_id"]

        counts = reaction_counts_by_outfit.get(oid, {"fire": 0, "clean": 0, "bold": 0})
        outfit["reaction_counts"] = counts
        outfit["total_reactions"] = sum(counts.values())

        outfit["creator"] = creator_profiles.get(creator_id, {
            "user_id": creator_id, "display_name": None, "avatar_url": None
        })

        item_ids = outfit.get("item_ids") or []
        outfit["cover_images"] = [
            cover_images_by_outfit_items[iid]
            for iid in item_ids[:2]
            if iid in cover_images_by_outfit_items
        ]
        outfit["image_count"] = len(item_ids)

        outfit["my_reaction"] = my_reactions.get(oid)
        outfit["is_saved"] = oid in my_saved
        outfit["is_following_creator"] = creator_id in my_following
        outfit["is_mine"] = creator_id == str(user_id)

        outfit["comment_count"] = 0
        outfit["share_count"] = 0

    if sort == "popular":
        outfits.sort(key=lambda o: o["total_reactions"], reverse=True)

    return SuccessResponse(data={
        "outfits": outfits,
        "pagination": {
            "total": total, "limit": limit, "offset": offset,
            "has_more": (offset + limit) < total
        }
    })
def _get_bulk_reaction_counts(outfit_ids: list[str]) -> dict[str, dict]:
    """Fetch reaction counts for multiple outfits in one query, grouped by outfit_id."""
    if not outfit_ids:
        return {}

    result = supabase.table("outfit_reactions")\
        .select("outfit_id, reaction")\
        .in_("outfit_id", outfit_ids)\
        .execute()

    counts_by_outfit: dict[str, dict] = {oid: {"fire": 0, "clean": 0, "bold": 0} for oid in outfit_ids}

    for row in (result.data or []):
        oid = row["outfit_id"]
        r = row["reaction"]
        if oid in counts_by_outfit and r in counts_by_outfit[oid]:
            counts_by_outfit[oid][r] += 1

    return counts_by_outfit
def _get_bulk_creator_profiles(user_ids: list[str]) -> dict[str, dict]:
    """
    Fetch creator display info in one bulk query (profiles: id, full_name).

    avatar_url is ALWAYS None on purpose (AV-7 privacy decision, 2026-10-25):
    the try-on avatar is the user's full-body photo, uploaded for a private
    try-on experience. Publishing an outfit is not consent to show that
    photo to other users. A public profile picture will be a separate
    feature with its own explicit upload and visibility rules. Flutter
    shows the creator's initials when avatar_url is null.
    """
    if not user_ids:
        return {}

    profiles_result = supabase.table("profiles")\
        .select("id, full_name")\
        .in_("id", user_ids)\
        .execute()

    creators = {}
    for row in (profiles_result.data or []):
        uid = row["id"]
        creators[uid] = {
            "user_id": uid,
            "display_name": row.get("full_name"),
            "avatar_url": None
        }

    return creators

def _get_bulk_item_images(item_ids: list[str]) -> dict[str, str]:
    """Fetch clean_image_url for multiple wardrobe items in one query."""
    if not item_ids:
        return {}

    result = supabase.table("closet_items")\
        .select("item_id, clean_image_url")\
        .in_("item_id", item_ids)\
        .execute()

    return {
        row["item_id"]: row["clean_image_url"]
        for row in (result.data or [])
    }


def _get_bulk_my_reactions(outfit_ids: list[str], user_id: str) -> dict[str, str]:
    """Fetch the current user's reaction (if any) on each outfit in one query."""
    if not outfit_ids:
        return {}

    result = supabase.table("outfit_reactions")\
        .select("outfit_id, reaction")\
        .eq("user_id", user_id)\
        .in_("outfit_id", outfit_ids)\
        .execute()

    return {row["outfit_id"]: row["reaction"] for row in (result.data or [])}


def _get_bulk_saved_status(outfit_ids: list[str], user_id: str) -> set[str]:
    """Fetch which of these outfits the current user has bookmarked, in one query."""
    if not outfit_ids:
        return set()

    result = supabase.table("saved_outfits")\
        .select("outfit_id")\
        .eq("user_id", user_id)\
        .in_("outfit_id", outfit_ids)\
        .execute()

    return {row["outfit_id"] for row in (result.data or [])}


def _get_bulk_following_status(creator_ids: list[str], user_id: str) -> set[str]:
    """Fetch which of these creators the current user follows, in one query."""
    if not creator_ids:
        return set()

    result = supabase.table("follows")\
        .select("following_id")\
        .eq("follower_id", user_id)\
        .in_("following_id", creator_ids)\
        .execute()

    return {row["following_id"] for row in (result.data or [])}
@router.post("/save/{outfit_id}", response_model=SuccessResponse)
async def save_outfit_bookmark(
    outfit_id: UUID,
    user_id: UUID = Depends(get_current_user_id)
):
    """Bookmark an outfit for later"""
    logger.info(f"Save outfit - outfit={outfit_id} user={user_id}")

    outfit_result = supabase.table("outfits")\
        .select("outfit_id")\
        .eq("outfit_id", str(outfit_id))\
        .single()\
        .execute()

    if not outfit_result.data:
        raise FitCheckException(
            "Outfit not found.", code="OUTFIT_NOT_FOUND", status_code=404
        )

    existing = supabase.table("saved_outfits")\
        .select("outfit_id")\
        .eq("user_id", str(user_id))\
        .eq("outfit_id", str(outfit_id))\
        .execute()

    if existing.data:
        return SuccessResponse(data={"message": "Already saved", "outfit_id": str(outfit_id)})

    result = supabase.table("saved_outfits")\
        .insert({
            "user_id": str(user_id),
            "outfit_id": str(outfit_id)
        })\
        .execute()

    if not result.data:
        raise FitCheckException(
            "Failed to save outfit.", code="DATABASE_ERROR", status_code=500
        )

    return SuccessResponse(data={"message": "Outfit saved", "outfit_id": str(outfit_id)})


@router.delete("/save/{outfit_id}", response_model=SuccessResponse)
async def remove_saved_outfit(
    outfit_id: UUID,
    user_id: UUID = Depends(get_current_user_id)
):
    """Remove a bookmarked outfit"""
    logger.info(f"Unsave outfit - outfit={outfit_id} user={user_id}")

    supabase.table("saved_outfits")\
        .delete()\
        .eq("user_id", str(user_id))\
        .eq("outfit_id", str(outfit_id))\
        .execute()

    return SuccessResponse(data={"message": "Outfit removed from saved", "outfit_id": str(outfit_id)})


@router.get("/saved", response_model=SuccessResponse)
async def get_saved_outfits(
    user_id: UUID = Depends(get_current_user_id),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Get outfits this user has bookmarked"""
    result = supabase.table("saved_outfits")\
        .select("outfit_id, created_at, outfits(outfit_id, user_id, overall_score, overall_grade, item_ids, occasion, name)", count="exact")\
        .eq("user_id", str(user_id))\
        .order("created_at", desc=True)\
        .range(offset, offset + limit - 1)\
        .execute()

    saved = result.data or []
    total = result.count or 0

    return SuccessResponse(data={
        "saved_outfits": saved,
        "pagination": {
            "total": total, "limit": limit, "offset": offset,
            "has_more": (offset + limit) < total
        }
    })