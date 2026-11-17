"""
Try-On v2 T2-1 B1 (2026-11-17): outfit slots - the ONE place that decides
which wardrobe category goes in which slot and which slots render.

Categories are the closet_items.category enum (shared.models.enums.Category):
top, bottom, shoes, bag, accessory, jewellery, headwear, scarf, belt, outerwear, dress.

Render rules (FASHN v1.6: tops / bottoms / one-pieces, chained top -> bottom):
    one_piece            renders alone (no top / bottom with it)
    top                  renders, first in the chain
    outerwear            renders as the upper piece ONLY when there's no top and no
                         one_piece; otherwise it goes in the tray (3-layer renders
                         are a T2-2 prototype)
    bottom               renders, after the upper piece
    shoes / bag / accessories   always the tray (never drawn on the avatar)
"""

# Display / canonical order of slots.
SLOT_ORDER = ("one_piece", "top", "outerwear", "bottom", "shoes", "bag", "accessories")

SLOTS = {
    "one_piece":   {"renders": True,  "max": 1, "excludes": ["top", "bottom"], "tray_when": [],
                    "categories": ["dress"]},
    "top":         {"renders": True,  "max": 1, "excludes": ["one_piece"], "tray_when": [],
                    "categories": ["top"]},
    "outerwear":   {"renders": True,  "max": 1, "excludes": [], "tray_when": ["top", "one_piece"],
                    "categories": ["outerwear"]},
    "bottom":      {"renders": True,  "max": 1, "excludes": ["one_piece"], "tray_when": [],
                    "categories": ["bottom"]},
    "shoes":       {"renders": False, "max": 1, "excludes": [], "tray_when": [],
                    "categories": ["shoes"]},
    "bag":         {"renders": False, "max": 1, "excludes": [], "tray_when": [],
                    "categories": ["bag"]},
    "accessories": {"renders": False, "max": 4, "excludes": [], "tray_when": [],
                    "categories": ["accessory", "jewellery", "headwear", "scarf", "belt"]},
}

# A piece may also be worn in these slots (an overshirt as outerwear, a light jacket as a top).
EXTRA_SLOTS = {"top": ["outerwear"], "outerwear": ["top"]}

# FASHN category for a slot that renders.
FASHN_CATEGORY = {"one_piece": "one-pieces", "top": "tops", "outerwear": "tops", "bottom": "bottoms"}

_CATEGORY_TO_SLOT = {c: s for s, d in SLOTS.items() for c in d["categories"]}


def slot_for_category(category) -> str | None:
    """Default slot for a wardrobe category (None = unknown)."""
    value = getattr(category, "value", category)
    return _CATEGORY_TO_SLOT.get(str(value or "").strip().lower())


def allowed_slots(category) -> list[str]:
    default = slot_for_category(category)
    return [default, *EXTRA_SLOTS.get(default, [])] if default else []


def slots_payload() -> dict:
    return {
        "slots": [{"slot": s, **{k: v for k, v in SLOTS[s].items()}} for s in SLOT_ORDER],
        "render_order": ["one_piece", "top", "outerwear", "bottom"],
    }