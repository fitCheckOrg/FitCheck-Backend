"""
E20 Share link (2026-11-04).

Public, read-only pages served by this API (no login):
    GET /p/{post_id}             HTML page: collage, caption, creator name + @handle,
                                 reaction total, "Open in FitCheck" + store buttons
    GET /p/{post_id}/og.png      1200x630 Open Graph image made from the collage only
    GET /u/{handle}              small public profile page (name, @handle, bio, post count)
    GET /.well-known/assetlinks.json            Android App Links (/p/*, /u/*)
    GET /.well-known/apple-app-site-association iOS Universal Links (/p/*, /u/*)

PRIVACY: never shows an avatar, a profile photo or any person photo - only the
clean garment cut-outs (same rule as the share card). Only public posts that
aren't under review; anything else is a friendly "not available" page (404).

Config (environment variables, all optional until we have a domain):
    PUBLIC_WEB_URL      base for share links, e.g. https://fitcheck.app
                        (default http://localhost:8000 for testing)
    ANDROID_PACKAGE     e.g. com.fitcheck.app
    ANDROID_SHA256      one or more SHA-256 cert fingerprints, comma separated
    IOS_APP_ID          TEAMID.bundle.id, e.g. ABCDE12345.com.fitcheck.app
    PLAY_STORE_URL / APP_STORE_URL   store buttons (hidden when not set)
"""

import html
import io
import logging
import os
import time

import requests
from PIL import Image, ImageDraw

from core.storage.supabase_client import supabase

logger = logging.getLogger(__name__)

try:                                     # pick up .env if python-dotenv is installed
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OG_SIZE = (1200, 630)
OG_BG = (247, 244, 238)                  # FitCheck cream
OG_CACHE_SECONDS = 3600
MAX_PIECES = 4
_og_cache: dict[str, tuple[float, bytes]] = {}


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v.strip() if v and v.strip() else default


def base_url() -> str:
    return (_env("PUBLIC_WEB_URL", "http://localhost:8000") or "").rstrip("/")


def share_url(post_id) -> str:
    return f"{base_url()}/p/{post_id}"


def profile_url(handle: str) -> str:
    return f"{base_url()}/u/{handle}"


# ---------------------------------------------------------------- data

def _public_post(post_id: str) -> dict | None:
    rows = supabase.table("outfits")\
        .select("outfit_id, user_id, item_ids, caption, name, category, occasion, is_public, under_review")\
        .eq("outfit_id", post_id).limit(1).execute().data
    row = rows[0] if rows else None
    if not row or not row.get("is_public") or row.get("under_review"):
        return None
    return row


def _piece_images(item_ids: list) -> list[str]:
    ids = [str(i) for i in (item_ids or [])][:MAX_PIECES]
    if not ids:
        return []
    rows = supabase.table("closet_items").select("item_id, clean_image_url")\
        .in_("item_id", ids).execute().data or []
    by_id = {str(r["item_id"]): r.get("clean_image_url") for r in rows}
    return [by_id[i] for i in ids if by_id.get(i)]


def _creator(user_id: str) -> dict:
    rows = supabase.table("profiles").select("full_name, handle").eq("id", user_id).limit(1).execute().data
    return rows[0] if rows else {}


def _reaction_total(post_id: str) -> int:
    return supabase.table("outfit_reactions").select("user_id", count="exact")\
        .eq("outfit_id", post_id).limit(1).execute().count or 0


# ---------------------------------------------------------------- html

_STYLE = """
body{margin:0;background:#F7F4EE;color:#1B1A17;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
main{max-width:520px;margin:0 auto;padding:28px 18px 40px}
.brand{font-family:Georgia,'Playfair Display',serif;font-size:22px;letter-spacing:.5px;margin-bottom:18px}
.card{background:#fff;border-radius:22px;padding:18px;box-shadow:0 6px 24px rgba(0,0,0,.06)}
.collage{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.collage img{width:100%;aspect-ratio:1/1;object-fit:contain;background:#F7F4EE;border-radius:14px}
.collage.one{grid-template-columns:1fr}
h1{font-family:Georgia,serif;font-size:22px;margin:16px 0 4px}
.by{color:#6E6A62;font-size:15px}
.meta{color:#6E6A62;font-size:14px;margin-top:10px}
.btn{display:block;text-align:center;text-decoration:none;border-radius:999px;padding:14px;margin-top:12px;font-weight:600}
.primary{background:#1B1A17;color:#fff}.ghost{border:1px solid #1B1A17;color:#1B1A17}
.stores{display:flex;gap:10px}.stores a{flex:1}
"""


def _page(title: str, body: str, og: dict | None = None) -> str:
    og = og or {}
    metas = "".join(
        f'<meta property="{html.escape(k)}" content="{html.escape(v)}">'
        for k, v in og.items() if v)
    twitter = '<meta name="twitter:card" content="summary_large_image">' if og.get("og:image") else ""
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)}</title>{metas}{twitter}<style>{_STYLE}</style></head>'
            f'<body><main><div class="brand">FitCheck</div>{body}</main></body></html>')


def _store_buttons() -> str:
    play, apple = _env("PLAY_STORE_URL"), _env("APP_STORE_URL")
    links = []
    if play:
        links.append(f'<a class="btn ghost" href="{html.escape(play)}">Get it on Google Play</a>')
    if apple:
        links.append(f'<a class="btn ghost" href="{html.escape(apple)}">Download on the App Store</a>')
    return f'<div class="stores">{"".join(links)}</div>' if links else ""


def not_found_page() -> str:
    return _page("Not available - FitCheck",
                 '<div class="card"><h1>This look isn\'t available</h1>'
                 '<div class="by">It may have been taken down or made private.</div></div>'
                 + _store_buttons())


def post_page(post_id: str) -> str | None:
    post = _public_post(str(post_id))
    if not post:
        return None
    pid = str(post["outfit_id"])
    images = _piece_images(post.get("item_ids"))
    creator = _creator(str(post["user_id"]))
    name = creator.get("full_name") or "A FitCheck creator"
    handle = creator.get("handle")
    title = post.get("caption") or post.get("name") or "A look on FitCheck"
    reactions = _reaction_total(pid)
    by = html.escape(name) + (f' &middot; @{html.escape(handle)}' if handle else "")
    tags = " &middot; ".join(html.escape(str(t).replace("_", " ").capitalize())
                             for t in (post.get("category"), post.get("occasion")) if t)
    grid = "".join(f'<img src="{html.escape(u)}" alt="">' for u in images)
    body = (f'<div class="card"><div class="collage{" one" if len(images) == 1 else ""}">{grid}</div>'
            f'<h1>{html.escape(title)}</h1><div class="by">by {by}</div>'
            f'<div class="meta">{reactions} reaction{"" if reactions == 1 else "s"}'
            f'{(" &middot; " + tags) if tags else ""}</div></div>'
            f'<a class="btn primary" href="{html.escape(share_url(pid))}">Open in FitCheck</a>'
            + _store_buttons())
    og = {
        "og:type": "website",
        "og:site_name": "FitCheck",
        "og:title": f"{title} - by {name}",
        "og:description": f"{reactions} reaction{'' if reactions == 1 else 's'} on FitCheck",
        "og:url": share_url(pid),
        "og:image": f"{share_url(pid)}/og.png",
    }
    return _page(f"{title} - FitCheck", body, og)


def profile_page(handle: str) -> str | None:
    h = str(handle or "").strip().lstrip("@").lower()
    rows = supabase.table("profiles").select("id, full_name, handle, bio")\
        .eq("handle", h).limit(1).execute().data if h else None
    if not rows:
        return None
    p = rows[0]
    posts = supabase.table("outfits").select("outfit_id", count="exact")\
        .eq("user_id", p["id"]).eq("is_public", True).eq("under_review", False)\
        .limit(1).execute().count or 0
    name = p.get("full_name") or f"@{p['handle']}"
    body = (f'<div class="card"><h1>{html.escape(name)}</h1>'
            f'<div class="by">@{html.escape(p["handle"])}</div>'
            + (f'<div class="meta">{html.escape(p["bio"])}</div>' if p.get("bio") else "")
            + f'<div class="meta">{posts} look{"" if posts == 1 else "s"} on FitCheck</div></div>'
            f'<a class="btn primary" href="{html.escape(profile_url(p["handle"]))}">Open in FitCheck</a>'
            + _store_buttons())
    og = {"og:type": "profile", "og:site_name": "FitCheck",
          "og:title": f"{name} (@{p['handle']}) on FitCheck",
          "og:description": p.get("bio") or f"{posts} looks on FitCheck",
          "og:url": profile_url(p["handle"])}
    return _page(f"{name} - FitCheck", body, og)


# ---------------------------------------------------------------- og image

def _fetch(url: str) -> Image.Image | None:
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGBA")
    except Exception:
        logger.warning("og image piece fetch failed: %s", url)
        return None


def og_image(post_id: str) -> bytes | None:
    pid = str(post_id)
    hit = _og_cache.get(pid)
    if hit and time.time() - hit[0] < OG_CACHE_SECONDS:
        return hit[1]
    post = _public_post(pid)
    if not post:
        return None

    canvas = Image.new("RGB", OG_SIZE, OG_BG)
    pieces = [im for im in (_fetch(u) for u in _piece_images(post.get("item_ids"))) if im]
    n = len(pieces)
    if n:
        pad = 40
        cols = n if n <= 4 else 4
        cell_w = (OG_SIZE[0] - pad * (cols + 1)) // cols
        cell_h = OG_SIZE[1] - pad * 2 - 40
        for i, im in enumerate(pieces[:cols]):
            im.thumbnail((cell_w, cell_h), Image.LANCZOS)
            x = pad + i * (cell_w + pad) + (cell_w - im.width) // 2
            y = pad + (cell_h - im.height) // 2
            canvas.paste(im, (x, y), im)
    ImageDraw.Draw(canvas).text((40, OG_SIZE[1] - 50), "FitCheck", fill=(27, 26, 23))

    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    data = buf.getvalue()
    _og_cache[pid] = (time.time(), data)
    return data


# ---------------------------------------------------------------- app links

def assetlinks() -> list | None:
    package, prints = _env("ANDROID_PACKAGE"), _env("ANDROID_SHA256")
    if not package or not prints:
        return None
    return [{
        "relation": ["delegate_permission/common.handle_all_urls"],
        "target": {"namespace": "android_app", "package_name": package,
                   "sha256_cert_fingerprints": [p.strip() for p in prints.split(",") if p.strip()]},
    }]


def apple_app_site_association() -> dict | None:
    app_id = _env("IOS_APP_ID")
    if not app_id:
        return None
    return {"applinks": {"apps": [], "details": [
        {"appIDs": [app_id], "appID": app_id, "paths": ["/p/*", "/u/*"],
         "components": [{"/": "/p/*"}, {"/": "/u/*"}]},
    ]}}