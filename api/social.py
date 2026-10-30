from fastapi import APIRouter, Query, Depends
from uuid import UUID
from shared.logger import get_logger
from shared.models.responses import SuccessResponse
from shared.exceptions import FitCheckException
from core.storage.supabase_client import supabase
import re
from pydantic import BaseModel
from typing import Literal
from datetime import datetime, timezone
from shared.dependencies import get_current_user_id
from shared.models.enums import DiscoverCategory



logger = get_logger(__name__)

router = APIRouter(prefix="/social", tags=["Social"])
# A6: five reactions, in display order. One reaction per user per post.
REACTION_TYPES = ("love", "fire", "clean", "bold", "wear")
VALID_REACTIONS = set(REACTION_TYPES)


def _empty_counts() -> dict:
    return {r: 0 for r in REACTION_TYPES}



class ReactRequest(BaseModel):
    reaction: str


MAX_CAPTION_LEN = 220


class PublishPostRequest(BaseModel):
    caption: str | None = None
    category: DiscoverCategory | None = None
    show_score: bool = True


class EditPostRequest(BaseModel):
    caption: str | None = None
    category: DiscoverCategory | None = None
    show_score: bool | None = None

@router.get("/health")
async def health():
    return { "service": "social", "version": "1.8.0", "status": "healthy" }


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
        .limit(1)\
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
            "following_id": str(target_user_id),
            "is_following": True,
            **_get_follow_counts(str(target_user_id))
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
        "is_following": True,
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
        "is_following": False,
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
            f"Invalid reaction. Must be one of: {', '.join(REACTION_TYPES)}.",
            code="INVALID_REACTION", status_code=400
        )

    _require_visible_outfit(str(outfit_id), str(user_id))

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


def _require_visible_outfit(outfit_id: str, user_id: str) -> None:
    """404 unless the outfit exists AND is public or belongs to the caller."""
    res = supabase.table("outfits")\
        .select("outfit_id, user_id, is_public")\
        .eq("outfit_id", outfit_id)\
        .limit(1)\
        .execute()
    row = res.data[0] if res.data else None
    if not row or not (row.get("is_public") or str(row.get("user_id")) == user_id):
        raise FitCheckException(
            "Outfit not found.", code="OUTFIT_NOT_FOUND", status_code=404
        )


def _get_reaction_counts(outfit_id: str) -> dict:
    """Shared helper - returns reaction counts by type for an outfit."""
    counts = _empty_counts()

    result = supabase.table("outfit_reactions")\
        .select("reaction")\
        .eq("outfit_id", outfit_id)\
        .execute()

    for row in (result.data or []):
        r = row.get("reaction")
        if r in counts:
            counts[r] += 1

    return {"reaction_counts": counts, "total_reactions": sum(counts.values())}


FEED_FIELDS = (
    "outfit_id, user_id, overall_score, overall_grade, "
    "item_ids, occasion, name, created_at, published_at, is_public, category, "
    "caption, ai_highlights, show_score"
)


@router.get("/feed", response_model=SuccessResponse)
async def get_discover_feed(
    user_id: UUID = Depends(get_current_user_id),
    sort: Literal["recent", "popular", "trending", "following"] = Query("recent"),
    category: DiscoverCategory | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """
    Browse public outfits from the community.
    sort=recent    -> newest posts first (by published_at)
    sort=popular   -> most reactions ever first (across ALL posts, not per page)
    sort=trending  -> most reactions in the last 7 days first, then newest
    sort=following -> public outfits only from users you follow, newest first
    """
    logger.info(f"Get feed - user={user_id} sort={sort} category={category}")

    feed_fields = FEED_FIELDS

    if sort in ("popular", "trending"):
        return _ranked_feed(sort, category, limit, offset, str(user_id))

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
            .order("published_at", desc=True, nullsfirst=False)\
            .range(offset, offset + limit - 1)

    else:
        query = supabase.table("outfits")\
            .select(feed_fields, count="exact")\
            .eq("is_public", True)\
            .order("published_at", desc=True, nullsfirst=False)\
            .range(offset, offset + limit - 1)

    if category:
        query = query.eq("category", category.value)

    result = query.execute()
    outfits = result.data or []
    total = result.count or 0

    _decorate_posts(outfits, str(user_id))

    return SuccessResponse(data={
        "outfits": outfits,
        "pagination": {
            "total": total, "limit": limit, "offset": offset,
            "has_more": (offset + limit) < total
        }
    })
def _ranked_feed(sort: str, category, limit: int, offset: int, user_id: str) -> SuccessResponse:
    """
    A8: popular / trending, ranked in the database across ALL public posts
    (view public_post_rank), then the page's posts are loaded and decorated.
    Ties: newest post first.
    """
    metric = "total_reactions" if sort == "popular" else "reactions_7d"

    rq = supabase.table("public_post_rank")\
        .select("outfit_id", count="exact")
    if category:
        rq = rq.eq("category", category.value)
    ranked = rq.order(metric, desc=True)\
        .order("published_at", desc=True, nullsfirst=False)\
        .range(offset, offset + limit - 1)\
        .execute()

    ids = [r["outfit_id"] for r in (ranked.data or [])]
    total = ranked.count or 0

    outfits = []
    if ids:
        rows = supabase.table("outfits").select(FEED_FIELDS)\
            .in_("outfit_id", ids).eq("is_public", True).execute().data or []
        by_id = {r["outfit_id"]: r for r in rows}
        outfits = [by_id[i] for i in ids if i in by_id]   # keep the ranked order

    _decorate_posts(outfits, user_id)

    data = {
        "outfits": outfits,
        "pagination": {
            "total": total, "limit": limit, "offset": offset,
            "has_more": (offset + limit) < total
        }
    }
    if sort == "trending":
        # No location data yet (profiles have no city) -> Flutter shows
        # "Trending this week".
        data["region_label"] = None
    return SuccessResponse(data=data)


def _decorate_posts(outfits: list[dict], user_id: str) -> None:
    """
    Turn raw public outfit rows into feed/post items, in place.
    Shared by GET /feed and GET /posts/{id} so both return the same shape.
    """
    outfit_ids = [o["outfit_id"] for o in outfits]
    creator_ids = list({o["user_id"] for o in outfits})
    all_item_ids = list({item_id for o in outfits for item_id in (o.get("item_ids") or [])})

    reaction_counts_by_outfit = _get_bulk_reaction_counts(outfit_ids)
    creator_profiles = _get_bulk_creator_profiles(creator_ids)
    item_details = _get_bulk_item_details(all_item_ids)
    my_reactions = _get_bulk_my_reactions(outfit_ids, user_id)
    my_saved = _get_bulk_saved_status(outfit_ids, user_id)
    my_following = _get_bulk_following_status(creator_ids, user_id)

    for outfit in outfits:
        oid = outfit["outfit_id"]
        creator_id = outfit["user_id"]

        counts = reaction_counts_by_outfit.get(oid, _empty_counts())
        outfit["reaction_counts"] = counts
        outfit["total_reactions"] = sum(counts.values())

        creator = dict(creator_profiles.get(creator_id, {
            "user_id": creator_id, "display_name": None, "avatar_url": None
        }))
        creator["is_following"] = creator_id in my_following
        outfit["creator"] = creator

        item_ids = outfit.get("item_ids") or []
        # Pieces in outfit order (A1). A piece the owner deleted from their
        # closet comes back as {item_id, deleted: true}.
        outfit["items"] = [
            item_details.get(iid, {"item_id": iid, "deleted": True})
            for iid in item_ids
        ]
        # Kept for older app builds: first two cut-out images.
        outfit["cover_images"] = [
            item_details[iid]["image_url"]
            for iid in item_ids[:2]
            if iid in item_details and item_details[iid].get("image_url")
        ]
        outfit["image_count"] = len(item_ids)

        outfit["my_reaction"] = my_reactions.get(oid)
        outfit["is_saved"] = oid in my_saved
        outfit["is_following_creator"] = creator_id in my_following
        outfit["is_mine"] = creator_id == user_id

        outfit["comment_count"] = 0
        outfit["share_count"] = 0

        _apply_score_visibility(outfit)


@router.get("/posts/{post_id}", response_model=SuccessResponse)
async def get_post(
    post_id: UUID,
    user_id: UUID = Depends(get_current_user_id)
):
    """
    A4: one post, same shape as a feed item.
    404 POST_NOT_FOUND if it doesn't exist or isn't public (deleted or
    unpublished) - for everyone, including the owner.
    """
    result = supabase.table("outfits")\
        .select(FEED_FIELDS)\
        .eq("outfit_id", str(post_id))\
        .eq("is_public", True)\
        .limit(1)\
        .execute()

    if not result.data:
        raise FitCheckException(
            "This post isn't available.", code="POST_NOT_FOUND", status_code=404
        )

    posts = result.data
    _decorate_posts(posts, str(user_id))
    return SuccessResponse(data=posts[0])


def _clean_caption(caption) -> str | None:
    text = " ".join(str(caption or "").split())
    if len(text) > MAX_CAPTION_LEN:
        raise FitCheckException(
            f"Caption must be {MAX_CAPTION_LEN} characters or fewer.",
            code="CAPTION_TOO_LONG", status_code=400
        )
    return text or None


def _own_outfit(outfit_id: str, user_id: str) -> dict:
    res = supabase.table("outfits")\
        .select("outfit_id, is_public, published_at")\
        .eq("outfit_id", outfit_id).eq("user_id", user_id)\
        .limit(1).execute()
    if not res.data:
        raise FitCheckException(
            "Outfit not found.", code="OUTFIT_NOT_FOUND", status_code=404
        )
    return res.data[0]


def _post_response(post_id: str, user_id: str) -> SuccessResponse:
    res = supabase.table("outfits").select(FEED_FIELDS)\
        .eq("outfit_id", post_id).limit(1).execute()
    posts = res.data or []
    _decorate_posts(posts, user_id)
    return SuccessResponse(data=posts[0])


@router.post("/posts/{post_id}", response_model=SuccessResponse)
async def publish_post(
    post_id: UUID,
    body: PublishPostRequest,
    user_id: UUID = Depends(get_current_user_id)
):
    """
    A5: publish one of your saved outfits to Discover (or re-publish it with
    new settings). post_id = the outfit's outfit_id. Returns the post (feed shape).
    """
    uid, pid = str(user_id), str(post_id)
    current = _own_outfit(pid, uid)
    updates = {
        "is_public": True,
        "caption": _clean_caption(body.caption),
        "category": body.category.value if body.category else None,
        "show_score": body.show_score,
    }
    if not current.get("is_public") or not current.get("published_at"):
        updates["published_at"] = datetime.now(timezone.utc).isoformat()

    supabase.table("outfits").update(updates)\
        .eq("outfit_id", pid).eq("user_id", uid).execute()
    logger.info(f"Publish post - outfit={pid} user={uid}")
    return _post_response(pid, uid)


@router.patch("/posts/{post_id}", response_model=SuccessResponse)
async def edit_post(
    post_id: UUID,
    body: EditPostRequest,
    user_id: UUID = Depends(get_current_user_id)
):
    """
    A5: edit a published post. Only the fields you send change.
    caption "" clears it. 404 POST_NOT_FOUND if not yours or not published.
    """
    uid, pid = str(user_id), str(post_id)
    current = _own_outfit(pid, uid)
    if not current.get("is_public"):
        raise FitCheckException(
            "This post isn't available.", code="POST_NOT_FOUND", status_code=404
        )

    sent = body.model_fields_set
    updates = {}
    if "caption" in sent:
        updates["caption"] = _clean_caption(body.caption)
    if "category" in sent:
        updates["category"] = body.category.value if body.category else None
    if "show_score" in sent and body.show_score is not None:
        updates["show_score"] = body.show_score

    if updates:
        supabase.table("outfits").update(updates)\
            .eq("outfit_id", pid).eq("user_id", uid).execute()
    return _post_response(pid, uid)


@router.delete("/posts/{post_id}", response_model=SuccessResponse)
async def delete_post(
    post_id: UUID,
    user_id: UUID = Depends(get_current_user_id)
):
    """
    A5: take a post off Discover. The outfit itself stays in your Lookbook.
    Removes the post's reactions and other users' bookmarks of it, and clears
    caption + published_at. Idempotent: deleting an unpublished outfit is 200.
    """
    uid, pid = str(user_id), str(post_id)
    _own_outfit(pid, uid)

    supabase.table("outfits").update({
        "is_public": False, "caption": None, "published_at": None,
    }).eq("outfit_id", pid).eq("user_id", uid).execute()

    reactions = supabase.table("outfit_reactions").delete()\
        .eq("outfit_id", pid).execute()
    saves = supabase.table("saved_outfits").delete()\
        .eq("outfit_id", pid).neq("user_id", uid).execute()

    logger.info(f"Delete post - outfit={pid} user={uid}")
    return SuccessResponse(data={
        "post_id": pid,
        "deleted": True,
        "reactions_removed": len(reactions.data or []),
        "saves_removed": len(saves.data or []),
    })


# Public "why this works" note (A3). The stored AI texts are written TO the
# owner ("your minimalist style", "complements your athletic build"), so they
# can reveal the owner's style settings or body. Only sentences that talk
# purely about the clothes are allowed into a public post.
_PRIVATE_WORDS = re.compile(
    r"\b(you|your|yours|you're|youre|yourself|"
    r"build|body|bodies|shape|figure|frame|physique|height|tall|short|petite|"
    r"weight|waist|hips?|shoulders?|chest|bust|torso|legs?|arms?|skin|"
    r"complexion|undertone|tone)\b",
    re.IGNORECASE,
)
MAX_PUBLIC_NOTE_SENTENCES = 2


def _public_note(highlights) -> str | None:
    clean = []
    for sentence in (highlights or []):
        text = str(sentence or "").strip()
        if text and not _PRIVATE_WORDS.search(text):
            clean.append(text)
        if len(clean) >= MAX_PUBLIC_NOTE_SENTENCES:
            break
    return " ".join(clean) or None


def _apply_score_visibility(outfit: dict) -> None:
    """
    A3: the creator chooses whether the score + note are shown on the post.
    show_score defaults to True. When False, score, grade and note are null.
    The raw ai_summary / ai_highlights / ai_personalization_note are never
    sent in social responses (see _public_note).
    """
    note = _public_note(outfit.pop("ai_highlights", None))
    show = outfit.get("show_score")
    show = True if show is None else bool(show)
    outfit["show_score"] = show
    outfit["ai_summary"] = note if show else None
    if not show:
        outfit["overall_score"] = None
        outfit["overall_grade"] = None


def _get_bulk_reaction_counts(outfit_ids: list[str]) -> dict[str, dict]:
    """Fetch reaction counts for multiple outfits in one query, grouped by outfit_id."""
    if not outfit_ids:
        return {}

    result = supabase.table("outfit_reactions")\
        .select("outfit_id, reaction")\
        .in_("outfit_id", outfit_ids)\
        .execute()

    counts_by_outfit: dict[str, dict] = {oid: _empty_counts() for oid in outfit_ids}

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

def _get_bulk_item_details(item_ids: list[str]) -> dict[str, dict]:
    """
    Fetch the public piece info for many wardrobe items in one query.
    Only the clean cut-out image and descriptive tags - nothing personal.
    """
    if not item_ids:
        return {}

    result = supabase.table("closet_items")\
        .select("item_id, clean_image_url, category, item_type, dominant_color")\
        .in_("item_id", item_ids)\
        .execute()

    return {
        row["item_id"]: {
            "item_id": row["item_id"],
            "image_url": row.get("clean_image_url"),
            "category": row.get("category"),
            "item_type": row.get("item_type"),
            "dominant_color": row.get("dominant_color"),
            "deleted": False,
        }
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

    _require_visible_outfit(str(outfit_id), str(user_id))

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
        .select("outfit_id, created_at, outfits(outfit_id, user_id, overall_score, overall_grade, item_ids, occasion, name, is_public)", count="exact")\
        .eq("user_id", str(user_id))\
        .order("created_at", desc=True)\
        .range(offset, offset + limit - 1)\
        .execute()

    # Never show someone else's outfit once it's no longer public.
    saved = [
        row for row in (result.data or [])
        if row.get("outfits") and (
            row["outfits"].get("is_public")
            or str(row["outfits"].get("user_id")) == str(user_id)
        )
    ]
    total = result.count or 0

    return SuccessResponse(data={
        "saved_outfits": saved,
        "pagination": {
            "total": total, "limit": limit, "offset": offset,
            "has_more": (offset + limit) < total
        }
    })