"""
FitCheck — Garment Region Geometry: Long-Sleeve Repeat Test

Before concluding "long sleeves are unreliable," rule out the
simpler explanation: run-to-run randomness. Tests the same 2
known long-sleeve items 3 times each — if the suspicious
identical-bounds pattern shows up consistently, that's real
evidence of a garment-type-specific issue. If it's inconsistent
run to run, the earlier finding was likely a coincidence from a
too-small sample, not a real pattern.

Run from inside FitCheckAI, venv activated:
    python garment_geometry_validation.py
"""

import sys
import json
import base64
import requests

sys.path.insert(0, ".")

from core.storage.supabase_client import supabase
from core.config.settings import settings
import openai

client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

REPEAT_TEST_ITEM_IDS = [
    "25989827-db9d-4d65-90b3-12f5357b0b44",  # long sleeve t-shirt
    "ba2e5aa1-bfb5-462f-aaad-fe378f658c42",  # long sleeve shirt
]
REPEATS_PER_ITEM = 3

GARMENT_REGION_PROMPT = """
IMPORTANT — coordinate convention: use "left" and "right" as YOU,
the viewer, see them looking directly at this image — NOT the
wearer's anatomical left/right.

First, describe 2-3 SPECIFIC visual details you observe in THIS
exact image.

THEN provide a structured representation. For EVERY region (torso
and each visible sleeve), determine both its horizontal extent
(left_percent, right_percent) AND vertical extent (top_percent,
bottom_percent) — measure each region INDEPENDENTLY based on what
you actually see, even if two regions end up with similar bounds.
Do not assume sleeves share the same vertical extent as the torso
unless you can genuinely observe that they do.

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


def check_suspicious_identical_bounds(rep: dict) -> str | None:
    torso = rep.get("torso_region", {})
    left = rep.get("left_sleeve_region") or {}
    right = rep.get("right_sleeve_region") or {}

    t_bounds = (torso.get("top_percent"), torso.get("bottom_percent"))
    l_bounds = (left.get("top_percent"), left.get("bottom_percent"))
    r_bounds = (right.get("top_percent"), right.get("bottom_percent"))

    if left and t_bounds == l_bounds:
        return "SUSPICIOUS: left_sleeve vertical bounds identical to torso"
    if right and t_bounds == r_bounds:
        return "SUSPICIOUS: right_sleeve vertical bounds identical to torso"
    return None


def main():
    summary = []

    for item_id in REPEAT_TEST_ITEM_IDS:
        result = supabase.table("closet_items")\
            .select("item_id, item_type, clean_image_url")\
            .eq("item_id", item_id)\
            .single()\
            .execute()

        if not result.data:
            print(f"Item {item_id} not found, skipping")
            continue

        item = result.data
        image_bytes = requests.get(item["clean_image_url"]).content

        suspicious_runs = 0

        for run_num in range(1, REPEATS_PER_ITEM + 1):
            print(f"\n{'='*60}")
            print(f"item: {item_id} ({item['item_type']}) — run {run_num}/{REPEATS_PER_ITEM}")
            print('='*60)

            rep = get_garment_geometry(image_bytes)

            if "error" in rep:
                print(f"  ERROR: {rep['error']}")
                continue

            print(json.dumps(rep, indent=2))

            flag = check_suspicious_identical_bounds(rep)
            if flag:
                suspicious_runs += 1
                print(f"\n  >>> {flag}")

        summary.append({
            "item_id": item_id,
            "item_type": item["item_type"],
            "suspicious_runs": suspicious_runs,
            "total_runs": REPEATS_PER_ITEM
        })

    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    for s in summary:
        print(f"  {s['item_id']} ({s['item_type']}): "
              f"{s['suspicious_runs']}/{s['total_runs']} runs suspicious")


if __name__ == "__main__":
    main()