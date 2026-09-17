"""
FitCheck — Garment Representation POC

This is the current, correct version, reflecting everything
confirmed tonight — NOT the version from earlier in the
investigation. Key differences from the first draft:
  - PIL/OpenCV width-profile candidate DROPPED — demonstrated
    unreliable across all 4 real test items, confirmed not
    fixable by tuning (the core "widest row = shoulder line"
    assumption doesn't hold for real garment photos)
  - GPT-4o Vision prompt now includes the handedness fix —
    confirmed via a real before/after test that this resolves a
    genuine left/right coordinate inconsistency found earlier
  - Schema expanded to the full structured representation the
    supervisor asked to evaluate (collar, torso, sleeves, hem)

Purpose right now: produce this representation for several real
wardrobe items and print it for manual inspection — NOT to wire
into GarmentPositioning, the database, or Flutter. Pure evidence-
gathering, per the explicit instruction to prove the representation
is useful before integrating anything.

Run from inside FitCheckAI, venv activated:
    python garment_investigation.py
"""

import sys
import io
import json
import base64
import requests
from PIL import Image

sys.path.insert(0, ".")

from core.storage.supabase_client import supabase
from core.config.settings import settings
import openai

# ---- CONFIG — paste real item_ids from your wardrobe here ----
TEST_ITEM_IDS = [
    #"25989827-db9d-4d65-90b3-12f5357b0b44",  # long sleeve t-shirt
    #"ba2e5aa1-bfb5-462f-aaad-fe378f658c42",  # long sleeve shirt
    #"ac7fffb7-2e3a-40f8-894a-0e85fc245373",  # polo shirt
    #"bb7763f6-f1df-4b26-9522-94299149bb5e",  # polo shirt
    # add 1 more — a genuinely messy photo (different background/
    # angle/lighting than the clean shots above) per the original
    # test-set recommendation, not yet covered by this set
    
    #"e398e450-a4be-447a-b981-5a57297c7e66",
    #"2ea4bb57-a358-4349-8aaf-204ded0772ab",
    #"6a53c00f-4f84-4e64-ad3b-5d2753504481",
    
    "cb13467e-585f-4a4f-82fe-bac29082424e",
    "52f5fd29-d817-49bb-a0c8-ba181ba47157",
    "ded59bd3-57a1-43b1-888a-867e4362d2d8"
]

client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

GARMENT_REPRESENTATION_PROMPT = """
IMPORTANT — coordinate convention: use "left" and "right" as YOU,
the viewer, see them looking directly at this image — NOT the
wearer's anatomical left/right. So "left" always means appearing
on the LEFT SIDE of the image as displayed, regardless of which
arm it would be if someone were wearing the garment facing you.
Do not switch conventions between images — always use
viewer's-left, viewer's-right.

First, describe 2-3 SPECIFIC visual details you actually observe
in THIS exact image — things that would differ between two
different garments of the same type. For example: the sleeve
angle/position, any visible fabric folds or asymmetry, exact
collar style, how far the sleeves extend relative to the torso,
or any distinguishing pattern/seam detail. When mentioning a
sleeve, explicitly say "the sleeve on the left side of the image"
or "the sleeve on the right side of the image" — never just "the
left sleeve" or "the right sleeve" alone, to stay unambiguous.

THEN, based on what you just observed in THIS image, provide a
structured garment representation. This is NOT asking for precise
pixel coordinates — rough proportional estimates, as percentages
of the image's total width/height, are expected and fine.

Return ONLY valid JSON matching this exact schema, no markdown,
no explanation outside the JSON:
{
  "observations": "the 2-3 specific details you described above",
  "garment_type": "brief description, e.g. 'long sleeve crew neck shirt'",
  "collar": {
    "top_percent": X,
    "center_x_percent": X
  },
  "torso_region": {
    "left_percent": X,
    "right_percent": X
  },
  "left_sleeve_region": {
    "left_percent": X,
    "right_percent": X
  },
  "right_sleeve_region": {
    "left_percent": X,
    "right_percent": X
  },
  "hem_percent": X
}

If the garment has no distinct collar, or sleeves are not visible
(e.g. a sleeveless item, or a bottom/dress cropped to not show
arms), use null for that specific field rather than guessing.
All percent values are 0-100, relative to image width (for
left_percent/right_percent/center_x_percent) or image height
(for top_percent/hem_percent).
"""


def get_garment_representation(image_bytes: bytes) -> dict:
    """Real GPT-4o Vision call, using the existing configured API key."""
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
                    {"type": "text", "text": GARMENT_REPRESENTATION_PROMPT}
                ]
            }],
            max_tokens=400
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e)}


def main():
    if not TEST_ITEM_IDS:
        print("Add real item_ids to TEST_ITEM_IDS before running.")
        return

    all_results = []

    for item_id in TEST_ITEM_IDS:
        print(f"\n{'='*60}")
        print(f"Testing item: {item_id}")
        print('='*60)

        result = supabase.table("closet_items")\
            .select("item_id, item_type, category, clean_image_url")\
            .eq("item_id", item_id)\
            .single()\
            .execute()

        if not result.data:
            print(f"  Item not found, skipping")
            continue

        item = result.data
        print(f"  item_type: {item['item_type']}")
        print(f"  category: {item['category']}")

        image_bytes = requests.get(item["clean_image_url"]).content

        representation = get_garment_representation(image_bytes)
        print(f"\n  -- Garment Representation --")
        print(f"  {json.dumps(representation, indent=2)}")

        all_results.append({
            "item_id": item_id,
            "item_type": item["item_type"],
            "representation": representation
        })

    print(f"\n\n{'='*60}")
    print("All results (for copy/paste into review):")
    print('='*60)
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()