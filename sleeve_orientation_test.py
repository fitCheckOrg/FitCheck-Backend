"""
FitCheck — Garment Region Geometry Test (v2)

Drops the categorical orientation field (proved unreliable — 
correctly labeled the long-sleeve shirt as "hanging" but also 
mislabeled the polo the same way, despite genuinely different 
vertical extents). Keeps pure geometric bounds (horizontal + 
vertical) as the only signal — matches the supervisor's call to 
treat geometry as the trustworthy layer, semantic labels as not.

Run from inside FitCheckAI, venv activated:
    python sleeve_orientation_test.py
"""

import sys
import json
import base64
import requests

sys.path.insert(0, ".")

from core.storage.supabase_client import supabase
from core.config.settings import settings
import openai

TEST_ITEM_IDS = [
    "25989827-db9d-4d65-90b3-12f5357b0b44",  # long-sleeve, full-height sleeves in prior test
    "ac7fffb7-2e3a-40f8-894a-0e85fc245373",  # polo, localized upper-mid sleeves in prior test
]

client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

GARMENT_REGION_PROMPT = """
IMPORTANT — coordinate convention: use "left" and "right" as YOU,
the viewer, see them looking directly at this image — NOT the
wearer's anatomical left/right.

First, describe 2-3 SPECIFIC visual details you observe in THIS
exact image.

THEN provide a structured representation. For EVERY region (torso
and each visible sleeve), determine both its horizontal extent
(left_percent, right_percent) AND vertical extent (top_percent,
bottom_percent) — where it starts and ends in each dimension, as
percentages of the full image.

Return ONLY valid JSON matching this exact schema, no markdown,
no explanation outside the JSON:
{
  "observations": "...",
  "garment_type": "...",
  "torso_region": {
    "left_percent": X, "right_percent": X,
    "top_percent": X, "bottom_percent": X
  },
  "left_sleeve_region": {
    "left_percent": X, "right_percent": X,
    "top_percent": X, "bottom_percent": X
  } or null,
  "right_sleeve_region": {
    "left_percent": X, "right_percent": X,
    "top_percent": X, "bottom_percent": X
  } or null,
  "hem_percent": X
}

All percent values are 0-100, relative to image width
(left_percent/right_percent) or image height (top_percent/
bottom_percent/hem_percent).
"""


def get_garment_geometry(image_bytes: bytes) -> dict:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    try:
        response = client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{b64}"
                    }},
                    {"type": "text", "text": GARMENT_REGION_PROMPT}
                ]
            }],
            max_tokens=500
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e)}


def main():
    for item_id in TEST_ITEM_IDS:
        print(f"\n{'='*60}")
        print(f"Testing item: {item_id}")
        print('='*60)

        result = supabase.table("closet_items")\
            .select("item_id, item_type, clean_image_url")\
            .eq("item_id", item_id)\
            .single()\
            .execute()

        if not result.data:
            print("  Item not found, skipping")
            continue

        item = result.data
        print(f"  item_type: {item['item_type']}")
        image_bytes = requests.get(item["clean_image_url"]).content

        representation = get_garment_geometry(image_bytes)
        print(f"\n{json.dumps(representation, indent=2)}")


if __name__ == "__main__":
    main()