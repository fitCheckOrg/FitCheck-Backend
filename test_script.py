"""
Validates the hybrid geometry extraction against the two known
failure/success cases from tonight — no new GPT call needed, using
the already-confirmed reliable horizontal boundaries directly.

Run: python hybrid_geometry_test.py
"""

import sys
import requests
sys.path.insert(0, ".")

from core.storage.supabase_client import supabase
from workers.wardrobe.geometry_extraction import build_component_geometry

# Real horizontal boundaries, confirmed reliable across multiple 
# runs tonight for these two items
TEST_CASES = [
    {
        "item_id": "25989827-db9d-4d65-90b3-12f5357b0b44",
        "label": "hanging-sleeve shirt (previously broke torso POC)",
        "left_sleeve": (0, 20), "right_sleeve": (80, 100), "torso": (20, 80)
    },
    {
        "item_id": "ac7fffb7-2e3a-40f8-894a-0e85fc245373",
        "label": "extended-sleeve polo (previously worked)",
        "left_sleeve": (0, 20), "right_sleeve": (80, 100), "torso": (20, 80)
    },
]

for case in TEST_CASES:
    print(f"\n{'='*60}")
    print(f"{case['item_id']} — {case['label']}")
    print('='*60)

    result = supabase.table("closet_items")\
        .select("clean_image_url").eq("item_id", case["item_id"]).single().execute()
    image_bytes = requests.get(result.data["clean_image_url"]).content

    torso = build_component_geometry(image_bytes, *case["torso"])
    left = build_component_geometry(image_bytes, *case["left_sleeve"])
    right = build_component_geometry(image_bytes, *case["right_sleeve"])

    print(f"  torso:       top={torso.top_percent}  bottom={torso.bottom_percent}")
    print(f"  left_sleeve: top={left.top_percent}  bottom={left.bottom_percent}")
    print(f"  right_sleeve: top={right.top_percent}  bottom={right.bottom_percent}")