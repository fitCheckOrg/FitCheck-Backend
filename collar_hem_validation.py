# """
# Collar + hem validation, deliberately constructed sample.

# Pulls a controlled mix (dark/light, short/long-sleeve) rather than
# whatever's first in the wardrobe. Records algorithm output, human
# judgment, and confidence as three SEPARATE fields per item — no
# collapsing "uncertain-then-confirmed" into a clean pass/fail.

# Run: python collar_hem_validation.py
# """

# import sys
# import io
# import json
# import requests
# import numpy as np
# import cv2
# from PIL import Image, ImageDraw

# sys.path.insert(0, ".")
# from core.storage.supabase_client import supabase

# ALPHA_THRESHOLD = 10


# def get_binary_mask(image_bytes):
#     image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
#     alpha = np.array(image)[:, :, 3]
#     return (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255, image


# def extract_collar_hem(binary_mask):
#     contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return None, None, None

#     main_contour = max(contours, key=cv2.contourArea)

#     topmost_idx = np.argmin(main_contour[:, 0, 1])
#     collar_point = tuple(int(v) for v in main_contour[topmost_idx][0])

#     lowest_idx = np.argmax(main_contour[:, 0, 1])
#     hem_point = tuple(int(v) for v in main_contour[lowest_idx][0])

#     return main_contour, collar_point, hem_point


# def visualize(image, contour, collar, hem):
#     canvas = image.convert("RGB").copy()
#     draw = ImageDraw.Draw(canvas)
#     pts = [(int(p[0][0]), int(p[0][1])) for p in contour]
#     draw.line(pts + [pts[0]], fill=(0, 100, 255), width=2)

#     if collar:
#         x, y = collar
#         draw.ellipse([x-10, y-10, x+10, y+10], fill=(0, 255, 255))
#         draw.text((x+12, y), "COLLAR candidate", fill=(0, 255, 255))
#     if hem:
#         x, y = hem
#         draw.ellipse([x-10, y-10, x+10, y+10], fill=(255, 0, 255))
#         draw.text((x+12, y), "HEM candidate", fill=(255, 0, 255))

#     return canvas


# def main():
#     # Deliberately weighted queries — dark vs light, short vs long
#     queries = [
#         {"label": "dark_short", "filters": {"dominant_color": "black", "item_type_contains": None}, "limit": 2},
#         {"label": "dark_long", "filters": {"dominant_color": "black"}, "limit": 2},
#         {"label": "light_short", "filters": {"dominant_color": "white"}, "limit": 2},
#         {"label": "light_long", "filters": {"dominant_color": "white"}, "limit": 1},
#     ]

#     all_items = []
#     seen_ids = set()

#     for q in queries:
#         query = supabase.table("closet_items")\
#             .select("item_id, item_type, dominant_color, clean_image_url")\
#             .eq("category", "top")\
#             .eq("is_archived", False)\
#             .eq("dominant_color", q["filters"]["dominant_color"])\
#             .limit(q["limit"])\
#             .execute()
#         for item in (query.data or []):
#             if item["item_id"] not in seen_ids:
#                 all_items.append(item)
#                 seen_ids.add(item["item_id"])

#     print(f"Pulled {len(all_items)} items for controlled validation set\n")

#     results = []

#     for item in all_items:
#         print(f"\n{'='*60}")
#         print(f"{item['item_id']} — {item['item_type']} ({item['dominant_color']})")
#         print('='*60)

#         image_bytes = requests.get(item["clean_image_url"]).content
#         mask, original = get_binary_mask(image_bytes)
#         h_img, w_img = mask.shape

#         contour, collar, hem = extract_collar_hem(mask)
#         if contour is None:
#             print("  No contour found")
#             continue

#         collar_pct = (round(collar[0]/w_img*100, 1), round(collar[1]/h_img*100, 1)) if collar else None
#         hem_pct = (round(hem[0]/w_img*100, 1), round(hem[1]/h_img*100, 1)) if hem else None

#         print(f"  ALGORITHM OUTPUT — collar: {collar_pct}  hem: {hem_pct}")

#         annotated = visualize(original, contour, collar, hem)
#         filename = f"collar_hem_{item['item_id'][:8]}.png"
#         annotated.save(filename)
#         print(f"  Saved: {filename}")
#         print(f"  >>> RECORD SEPARATELY: human_judgment=?  confidence=?")

#         results.append({
#             "item_id": item["item_id"],
#             "item_type": item["item_type"],
#             "dominant_color": item["dominant_color"],
#             "algorithm_output": {"collar": collar_pct, "hem": hem_pct},
#             "human_judgment": None,       # fill in after viewing PNG
#             "confidence": None,           # "confident" / "uncertain"
#             "image_file": filename
#         })

#     with open("collar_hem_results.json", "w") as f:
#         json.dump(results, f, indent=2)
#     print(f"\n\nSaved results template to collar_hem_results.json — "
#           f"fill in human_judgment + confidence for each item after "
#           f"reviewing the PNGs.")


# if __name__ == "__main__":
#     main()

"""
Collar + hem validation, GENUINELY controlled sample this time.

Fix from previous version: dark_short/dark_long were both pulling
identical populations (color filter only, sleeve-length filter was
dead code). This version actually filters item_type text for
"long sleeve" presence/absence, since item_type is confirmed free
text (not a controlled enum) — ilike pattern matching is the real
mechanism, not a fake label.

Run: python collar_hem_validation.py
"""

import sys
import io
import json
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw

sys.path.insert(0, ".")
from core.storage.supabase_client import supabase

ALPHA_THRESHOLD = 10


def get_binary_mask(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    alpha = np.array(image)[:, :, 3]
    return (alpha > ALPHA_THRESHOLD).astype(np.uint8) * 255, image


def extract_collar_hem(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    main_contour = max(contours, key=cv2.contourArea)

    topmost_idx = np.argmin(main_contour[:, 0, 1])
    collar_point = tuple(int(v) for v in main_contour[topmost_idx][0])

    lowest_idx = np.argmax(main_contour[:, 0, 1])
    hem_point = tuple(int(v) for v in main_contour[lowest_idx][0])

    return main_contour, collar_point, hem_point


def visualize(image, contour, collar, hem):
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    pts = [(int(p[0][0]), int(p[0][1])) for p in contour]
    draw.line(pts + [pts[0]], fill=(0, 100, 255), width=2)

    if collar:
        x, y = collar
        draw.ellipse([x-10, y-10, x+10, y+10], fill=(0, 255, 255))
        draw.text((x+12, y), "COLLAR candidate", fill=(0, 255, 255))
    if hem:
        x, y = hem
        draw.ellipse([x-10, y-10, x+10, y+10], fill=(255, 0, 255))
        draw.text((x+12, y), "HEM candidate", fill=(255, 0, 255))

    return canvas


def fetch_stratified(color: str, want_long_sleeve: bool, limit: int):
    """Genuinely different queries this time — color via eq(), 
    sleeve length via ilike() text matching on item_type, which 
    is confirmed free text, not an enum."""
    query = supabase.table("closet_items")\
        .select("item_id, item_type, dominant_color, clean_image_url")\
        .eq("category", "top")\
        .eq("is_archived", False)\
        .eq("dominant_color", color)

    if want_long_sleeve:
        query = query.ilike("item_type", "%long sleeve%")
    else:
        query = query.not_.ilike("item_type", "%long sleeve%")

    result = query.limit(limit).execute()
    return result.data or []


def main():
    all_items = []
    seen_ids = set()

    strata = [
        {"label": "dark_short", "color": "black", "long_sleeve": False, "limit": 2},
        {"label": "dark_long", "color": "black", "long_sleeve": True, "limit": 2},
        {"label": "light_short", "color": "white", "long_sleeve": False, "limit": 2},
        {"label": "light_long", "color": "white", "long_sleeve": True, "limit": 2},
    ]

    for stratum in strata:
        items = fetch_stratified(stratum["color"], stratum["long_sleeve"], stratum["limit"])
        print(f"{stratum['label']}: pulled {len(items)} items — "
              f"{[i['item_type'] for i in items]}")
        for item in items:
            if item["item_id"] not in seen_ids:
                item["stratum"] = stratum["label"]
                all_items.append(item)
                seen_ids.add(item["item_id"])

    print(f"\nTotal unique items in validation set: {len(all_items)}\n")

    results = []

    for item in all_items:
        print(f"\n{'='*60}")
        print(f"{item['item_id']} — {item['item_type']} ({item['dominant_color']}) [{item['stratum']}]")
        print('='*60)

        image_bytes = requests.get(item["clean_image_url"]).content
        mask, original = get_binary_mask(image_bytes)
        h_img, w_img = mask.shape

        contour, collar, hem = extract_collar_hem(mask)
        if contour is None:
            print("  No contour found")
            continue

        collar_pct = (round(collar[0]/w_img*100, 1), round(collar[1]/h_img*100, 1)) if collar else None
        hem_pct = (round(hem[0]/w_img*100, 1), round(hem[1]/h_img*100, 1)) if hem else None

        print(f"  ALGORITHM OUTPUT — collar: {collar_pct}  hem: {hem_pct}")

        annotated = visualize(original, contour, collar, hem)
        filename = f"collar_hem_{item['item_id'][:8]}.png"
        annotated.save(filename)
        print(f"  Saved: {filename}")
        print(f"  >>> RECORD SEPARATELY: human_judgment=?  confidence=?  failure_category=?")

        results.append({
            "item_id": item["item_id"],
            "item_type": item["item_type"],
            "dominant_color": item["dominant_color"],
            "stratum": item["stratum"],
            "algorithm_output": {"collar": collar_pct, "hem": hem_pct},
            "human_judgment": None,
            "confidence": None,
            "failure_category": None,
            "image_file": filename
        })

    with open("collar_hem_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nSaved results template to collar_hem_results.json")


if __name__ == "__main__":
    main()