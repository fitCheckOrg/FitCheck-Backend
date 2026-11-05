"""
D17 Recreate with my closet (2026-11-03).

POST /api/stylist/recreate {"post_id"}
  -> {items, pieces, matched, total, missing, overall_score, overall_grade}

For each piece of the post (in the post's order) find the closest piece in MY
active closet (not archived). Each of my pieces is used at most once.

    match      when                                                   points
    exact      same category + same type + same main colour            100
    close      same category + same type + colour in my piece's colours 90
    same_type  same category + same type, other colour                  75
    similar    same category + same colour, other type                  60
    category   same category only                                       45
    missing    nothing in that category                                  0

overall_score = average points over the post's pieces (0-100), graded on the
usual scale. It says how close your version is - it is NOT the stylist's
outfit score. Pieces the creator deleted from their closet are skipped.

Each call counts one recreate per person per post per day (post_recreates),
used by E19 insights. Recreating your own post is allowed but not counted.
"""

import logging

from postgrest.exceptions import APIError

from core.storage.supabase_client import supabase
from shared.exceptions import FitCheckException
from pipelines.safety_pipeline import _visible_post

logger = logging.getLogger(__name__)

MIN_WARDROBE = 5
POINTS = {"exact": 100, "close": 90, "same_type": 75, "similar": 60, "category": 45}
WARDROBE_FIELDS = (
    "item_id, clean_image_url, original_image_url, category, item_type, colors, "
    "dominant_color, secondary_color, dominant_hex, pattern, style, seasons, occasions, "
    "brand, material, favorite, times_worn, last_worn, is_archived, created_at"
)


def _bad(code: str, message: str, status: int = 400) -> FitCheckException:
    return FitCheckException(message, code=code, status_code=status)


def _norm(v) -> str:
    return str(v or "").strip().lower()


def _grade(score: int) -> str:
    if score >= 95:
        return "A+"
    if score >= 90:
        return "A"
    if score >= 85:
        return "B+"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "Needs Work"


def _match_kind(target: dict, mine: dict) -> str | None:
    if _norm(target.get("category")) != _norm(mine.get("category")):
        return None
    same_type = _norm(target.get("item_type")) and _norm(target.get("item_type")) == _norm(mine.get("item_type"))
    colour = _norm(target.get("dominant_color"))
    my_colours = {_norm(c) for c in (mine.get("colors") or [])} | {
        _norm(mine.get("dominant_color")), _norm(mine.get("secondary_color"))}
    same_colour = bool(colour) and colour == _norm(mine.get("dominant_color"))
    colour_in = bool(colour) and colour in my_colours
    if same_type and same_colour:
        return "exact"
    if same_type and colour_in:
        return "close"
    if same_type:
        return "same_type"
    if colour_in:
        return "similar"
    return "category"


def _count_recreate(post_id: str, me: str) -> None:
    try:
        supabase.table("post_recreates").insert({"post_id": post_id, "user_id": me}).execute()
    except APIError as exc:
        if getattr(exc, "code", None) != "23505":          # already counted today
            logger.warning("recreate count failed post=%s: %s", post_id, exc)
    except Exception:
        logger.exception("recreate count failed post=%s", post_id)


def recreate(post_id, user_id) -> dict:
    pid, me = str(post_id), str(user_id)
    post = _visible_post(pid, me)                                    # 404 POST_NOT_FOUND
    post_row = supabase.table("outfits").select("outfit_id, user_id, item_ids")\
        .eq("outfit_id", pid).limit(1).execute().data[0]

    closet = supabase.table("closet_items").select(WARDROBE_FIELDS)\
        .eq("user_id", me).eq("is_archived", False).execute().data or []
    if len(closet) < MIN_WARDROBE:
        raise _bad("INSUFFICIENT_WARDROBE",
                   f"Add at least {MIN_WARDROBE} pieces to your closet to recreate looks.")

    ids = [str(i) for i in (post_row.get("item_ids") or [])]
    found = supabase.table("closet_items")\
        .select("item_id, category, item_type, dominant_color, clean_image_url")\
        .in_("item_id", ids).execute().data if ids else []
    by_id = {str(r["item_id"]): r for r in (found or [])}
    targets = [by_id[i] for i in ids if i in by_id]                # deleted pieces skipped

    used: set[str] = set()
    pieces, items, missing, points = [], [], [], 0
    for t in targets:
        best, best_kind = None, None
        for mine in closet:
            if str(mine["item_id"]) in used:
                continue
            kind = _match_kind(t, mine)
            if kind and (best_kind is None or POINTS[kind] > POINTS[best_kind]):
                best, best_kind = mine, kind
                if kind == "exact":
                    break
        target_info = {"item_id": str(t["item_id"]), "category": t.get("category"),
                       "item_type": t.get("item_type"), "dominant_color": t.get("dominant_color"),
                       "image_url": t.get("clean_image_url")}
        if best:
            used.add(str(best["item_id"]))
            items.append(best)
            points += POINTS[best_kind]
            pieces.append({"post_piece": target_info, "match": best_kind, "my_item": best})
        else:
            missing.append({"category": t.get("category"), "item_type": t.get("item_type"),
                            "dominant_color": t.get("dominant_color")})
            pieces.append({"post_piece": target_info, "match": "missing", "my_item": None})

    total = len(targets)
    score = round(points / total) if total else 0
    if str(post["user_id"]) != me:
        _count_recreate(pid, me)
    logger.info("recreate post=%s user=%s matched=%s/%s score=%s", pid, me, len(items), total, score)
    return {
        "post_id": pid,
        "items": items,
        "pieces": pieces,
        "matched": len(items),
        "total": total,
        "missing": missing,
        "overall_score": score,
        "overall_grade": _grade(score),
    }