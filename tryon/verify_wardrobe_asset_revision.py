"""tryon/verify_wardrobe_asset_revision.py"""
import requests

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"

if __name__ == "__main__":
    resp = requests.get(
        f"http://127.0.0.1:8000/api/wardrobe/{GOLDEN_GARMENT_ID}?user_id={AVATAR_USER_ID}"
    ).json()
    print(resp["data"].get("asset_revision"))