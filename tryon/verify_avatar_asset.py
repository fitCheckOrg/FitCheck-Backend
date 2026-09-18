"""tryon/verify_avatar_asset.py — golden-avatar validation for Task 1.1"""
from workers.avatar.asset_builder import build_avatar_asset

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"

if __name__ == "__main__":
    asset = build_avatar_asset(AVATAR_USER_ID)
    print(asset.model_dump_json(indent=2))