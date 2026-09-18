"""tryon/verify_garment_asset.py — golden-garment validation for Task 1.2"""
from workers.wardrobe.asset_builder import build_garment_asset

GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"

if __name__ == "__main__":
    asset = build_garment_asset(GOLDEN_GARMENT_ID, AVATAR_USER_ID)
    print(asset.model_dump_json(indent=2))