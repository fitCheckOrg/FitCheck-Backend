"""
D14 Search (2026-11-03).

GET /api/social/search?q=&type=posts|creators|styles&cursor=&limit=
GET /api/social/search/suggestions

posts     every word of q must appear (as a word start) in the post's caption,
          name, category, occasion or a piece's colour / type / category.
          "black blazer" matches a post with a black piece and a blazer.
          Ranked by how many words hit the pieces + caption, then reactions in
          the last 7 days, then newest. Searches the newest POST_CANDIDATES posts.
creators  handle starts with q (@ optional) or any word of the name starts with q.
          Most followers first. Each row = user + counts {followers}.
styles    the Discover categories whose name / label starts with q, with post_count.

Never: posts under review, blocked / hidden people, posts marked not interested.
"""

import base64
import logging
import re

from core.storage.supabase_client import supabase
from shared.exceptions import FitCheckException
from shared.models.enums import DiscoverCategory
from pipelines.safety_pipeline import excluded_user_ids, not_interested_post_ids, not_in

logger = logging.getLogger(__name__)

MIN_QUERY = 2
MAX_QUERY = 60
POST_CANDIDATES = 200
CREATOR_CANDIDATES = 200
CHUNK = 80
DEFAULT_LIMIT = 20
MAX_LIMIT = 50
TRENDING_STYLES = 6
TOP_COLOURS = 12

COLOUR_HEX = {
    "black": "#111111", "white": "#FFFFFF", "grey": "#8E8E8E", "gray": "#8E8E8E",
    "navy": "#1F2A44", "blue": "#2F6FD6", "light blue": "#9CC3E6", "red": "#C8102E",
    "burgundy": "#7A1E2C", "pink": "#F2A7C3", "purple": "#6B3FA0", "green": "#2E7D32",
    "olive": "#6B6B2A", "khaki": "#BDB07A", "beige": "#E8DCC4", "cream": "#F3EBDD",
    "brown": "#6D4C35", "tan": "#C8A27C", "camel": "#C19A6B", "yellow": "#F2C230",
    "mustard": "#D4A017", "orange": "#E8772E", "gold": "#C9A227", "silver": "#C0C0C0",
    "denim": "#4A6A8A", "camouflage": "#6F7350", "multicolor": None, "multicolour": None,
}


def _bad(code: str, message: str, status: int = 400) -> FitCheckException:
    return FitCheckException(message, code=code, status_code=status)


def _clean_query(raw) -> str:
    q = " ".join(str(raw or "").strip().split())[:MAX_QUERY]
    if len(q.lstrip("@")) < MIN_QUERY:
        raise _bad("QUERY_TOO_SHORT", f"Type at least {MIN_QUERY} characters.")
    return q


def _words(text) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").replace("_", " ").lower())


def _encode_offset(n: int) -> str:
    return base64.urlsafe_b64encode(f"o|{n}".encode()).decode()


def _decode_offset(cursor: str | None) -> int:
    if not cursor:
        return 0
    try:
        kind, n = base64.urlsafe_b64decode(cursor.encode()).decode().split("|", 1)
        if kind != "o":
            raise ValueError
        return max(0, int(n))
    except Exception:
        raise _bad("INVALID_CURSOR", "That page cursor isn't valid. Start from the first page.")


def _page(items: list, cursor, limit) -> tuple[list, str | None]:
    limit = max(1, min(int(limit), MAX_LIMIT))
    offset = _decode_offset(cursor)
    page = items[offset:offset + limit]
    nxt = _encode_offset(offset + limit) if offset + limit < len(items) else None
    return page, nxt


def _chunks(ids: list[str]):
    for i in range(0, len(ids), CHUNK):
        yield ids[i:i + CHUNK]


def _label(value: str) -> str:
    return value.replace("_", " ").capitalize()


# ---------------------------------------------------------------- posts

def _pieces(item_ids: list[str]) -> dict[str, dict]:
    out = {}
    for part in _chunks(list({str(i) for i in item_ids if i})):
        for r in supabase.table("closet_items")\
                .select("item_id, dominant_color, item_type, category")\
                .in_("item_id", part).execute().data or []:
            out[str(r["item_id"])] = r
    return out


def _recent(post_ids: list[str]) -> dict[str, int]:
    out = {}
    for part in _chunks(post_ids):
        for r in supabase.table("public_post_rank").select("outfit_id, reactions_7d")\
                .in_("outfit_id", part).execute().data or []:
            out[str(r["outfit_id"])] = int(r.get("reactions_7d") or 0)
    return out


def search_posts(q: str, viewer: str, cursor, limit) -> dict:
    from api.social import FEED_FIELDS, _decorate_posts     # avoid circular import

    terms = _words(q)
    if not terms:
        return {"items": [], "next_cursor": None}

    query = supabase.table("outfits").select(FEED_FIELDS)\
        .eq("is_public", True).eq("under_review", False)
    query = not_in(query, "user_id", excluded_user_ids(viewer))
    query = not_in(query, "outfit_id", not_interested_post_ids(viewer))
    rows = query.order("published_at", desc=True, nullsfirst=False)\
        .limit(POST_CANDIDATES).execute().data or []

    pieces = _pieces([i for r in rows for i in (r.get("item_ids") or [])])
    scored = []
    for r in rows:
        piece_words, strong = set(), set()
        for iid in r.get("item_ids") or []:
            p = pieces.get(str(iid))
            if p:
                piece_words.update(_words(p.get("dominant_color")))
                piece_words.update(_words(p.get("item_type")))
                piece_words.update(_words(p.get("category")))
        strong = piece_words | set(_words(r.get("caption")))
        everything = strong | set(_words(r.get("name"))) | set(_words(r.get("category"))) \
            | set(_words(r.get("occasion")))

        def hit(term, pool):
            return any(w.startswith(term) for w in pool)

        if all(hit(t, everything) for t in terms):
            scored.append((sum(hit(t, strong) for t in terms), r))

    recent = _recent([str(r["outfit_id"]) for _, r in scored])
    scored.sort(key=lambda sr: (sr[0], recent.get(str(sr[1]["outfit_id"]), 0),
                                str(sr[1].get("published_at") or "")), reverse=True)
    page, nxt = _page([r for _, r in scored], cursor, limit)
    _decorate_posts(page, viewer)
    return {"items": page, "next_cursor": nxt}


# ---------------------------------------------------------------- creators

def search_creators(q: str, viewer: str, cursor, limit) -> dict:
    from pipelines.users_pipeline import build_users, _search_term   # avoid circular import

    term = _search_term(q)
    if len(term) < MIN_QUERY:
        return {"items": [], "next_cursor": None}
    rows = supabase.table("profiles").select("id")\
        .or_(f'handle.ilike."{term}*",full_name.ilike."{term}*",full_name.ilike."* {term}*"')\
        .limit(CREATOR_CANDIDATES).execute().data or []
    excluded = excluded_user_ids(viewer)
    ids = [str(r["id"]) for r in rows if str(r["id"]) not in excluded]

    followers: dict[str, int] = {i: 0 for i in ids}
    for part in _chunks(ids):
        for f in supabase.table("follows").select("following_id")\
                .in_("following_id", part).execute().data or []:
            fid = str(f["following_id"])
            followers[fid] = followers.get(fid, 0) + 1

    ids.sort(key=lambda i: followers.get(i, 0), reverse=True)
    page_ids, nxt = _page(ids, cursor, limit)
    people = build_users(page_ids, viewer)
    items = []
    for i in page_ids:
        if i in people:
            items.append({**people[i], "counts": {"followers": followers.get(i, 0)}})
    return {"items": items, "next_cursor": nxt}


# ---------------------------------------------------------------- styles

def _style_counts() -> dict[str, int]:
    counts = {c.value: 0 for c in DiscoverCategory}
    for c in DiscoverCategory:
        counts[c.value] = supabase.table("outfits").select("outfit_id", count="exact")\
            .eq("is_public", True).eq("under_review", False).eq("category", c.value)\
            .limit(1).execute().count or 0
    return counts


def search_styles(q: str, cursor, limit) -> dict:
    terms = _words(q)
    matches = [c.value for c in DiscoverCategory
               if all(any(w.startswith(t) for w in _words(c.value)) for t in terms)]
    counts = _style_counts() if matches else {}
    items = [{"tag": v, "label": _label(v), "post_count": counts.get(v, 0)} for v in matches]
    items.sort(key=lambda s: s["post_count"], reverse=True)
    page, nxt = _page(items, cursor, limit)
    return {"items": page, "next_cursor": nxt}


# ---------------------------------------------------------------- entry points

def search(raw_q, type_: str, viewer_id, cursor=None, limit: int = DEFAULT_LIMIT) -> dict:
    q = _clean_query(raw_q)
    viewer = str(viewer_id)
    if type_ == "creators":
        return search_creators(q, viewer, cursor, limit)
    if type_ == "styles":
        return search_styles(q, cursor, limit)
    return search_posts(q, viewer, cursor, limit)


def suggestions() -> dict:
    """Trending styles (most reactions this week, then most posts) + the colours people wear."""
    week: dict[str, int] = {c.value: 0 for c in DiscoverCategory}
    rows = supabase.table("public_post_rank").select("category, reactions_7d").execute().data or []
    for r in rows:
        if r.get("category") in week:
            week[r["category"]] += int(r.get("reactions_7d") or 0)
    posts = _style_counts()
    ranked = sorted(week, key=lambda v: (week[v], posts.get(v, 0)), reverse=True)
    trending = [v for v in ranked if posts.get(v, 0) > 0][:TRENDING_STYLES] or ranked[:TRENDING_STYLES]

    recent_posts = supabase.table("outfits").select("item_ids")\
        .eq("is_public", True).eq("under_review", False)\
        .order("published_at", desc=True, nullsfirst=False).limit(POST_CANDIDATES).execute().data or []
    pieces = _pieces([i for r in recent_posts for i in (r.get("item_ids") or [])])
    colour_count: dict[str, int] = {}
    for p in pieces.values():
        name = (p.get("dominant_color") or "").strip().lower()
        if name:
            colour_count[name] = colour_count.get(name, 0) + 1
    colours = sorted(colour_count, key=colour_count.get, reverse=True)[:TOP_COLOURS]

    return {
        "trending_styles": [{"tag": v, "label": _label(v)} for v in trending],
        "colours": [{"name": c.title(), "hex": COLOUR_HEX.get(c)} for c in colours],
    }