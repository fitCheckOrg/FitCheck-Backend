"""
Try-On v2 T2-1 B5 (2026-11-18): fresh links for private images.

POST /api/media/sign  {keys: [key | url, ...]}   (1-50)
    -> {items: [{key, url, expires_at}], denied: [key, ...]}

    * Private keys are signed ONLY when they belong to the caller:
        tryons/...                       a render in tryon_renders or a saved look of theirs
        {avatars}/{me}/original|processed/...
        {wardrobe}/{me}/original/...
    * Public keys (cut-outs, profile photos) come back with their plain url and
      expires_at null - they never expire.
    * Anything else is listed in `denied` (never says whether the file exists).
"""

from core.storage.s3 import storage, S3Folders
from core.storage.supabase_client import supabase
from shared.exceptions import FitCheckException

MAX_KEYS = 50


def _bad(code: str, message: str, status: int = 400) -> FitCheckException:
    return FitCheckException(message, code=code, status_code=status)


def _owned_renders(uid: str, keys: list[str]) -> set[str]:
    if not keys:
        return set()
    owned = {r["s3_key"] for r in supabase.table("tryon_renders").select("s3_key")
             .eq("user_id", uid).in_("s3_key", keys).execute().data or []}
    rest = [k for k in keys if k not in owned]
    if rest:
        owned |= {r["tryon_key"] for r in supabase.table("tryon_looks").select("tryon_key")
                  .eq("user_id", uid).in_("tryon_key", rest).execute().data or []}
    return owned


def sign_keys(user_id, keys) -> dict:
    uid = str(user_id)
    if not isinstance(keys, list) or not keys:
        raise _bad("KEYS_REQUIRED", "Send the keys to sign.")
    if len(keys) > MAX_KEYS:
        raise _bad("TOO_MANY_KEYS", f"Sign at most {MAX_KEYS} keys at a time.")
    wanted = list(dict.fromkeys(storage.key_from_url(str(k)) or str(k) for k in keys))

    renders = [k for k in wanted if storage.is_private(k) and k.startswith(S3Folders.TRYON + "/")]
    owned_renders = _owned_renders(uid, renders)

    items, denied = [], []
    for key in wanted:
        if not storage.is_private(key):
            if storage.key_from_url(key) == key and "/" in key:
                items.append(storage.signed(key))
            else:
                denied.append(key)
            continue
        if key.startswith(S3Folders.TRYON + "/"):
            mine = key in owned_renders
        else:                                   # {avatars|wardrobe}/{owner}/...
            folder = S3Folders.AVATARS if key.startswith(S3Folders.AVATARS + "/") else S3Folders.WARDROBE
            mine = key[len(folder) + 1:].split("/", 1)[0] == uid
        if mine:
            items.append(storage.signed(key))
        else:
            denied.append(key)
    return {"items": items, "denied": denied}