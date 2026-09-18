"""
Verifies whether the "wrong winner" candidates from the Layer 2
gap diagnostic coincide with already-known underarm landmarks —
own side or opposite side — rather than trusting pattern-matching
from memory.

Pure verification. No selection rule. No threshold. Just: how far
is each wrong winner from each known underarm coordinate?

Run: python wrong_winner_landmark_check.py
"""

import numpy as np

# All coordinates below are DIRECTLY from confirmed tonight's data —
# underarms sourced from earlier session outputs, wrong winners
# copied exactly from the gap diagnostic just run.

UNDERARMS = {
    "polo": {"LEFT": (106, 223), "RIGHT": (357, 222)},
    "hanging_shirt": {"LEFT": (109, 258), "RIGHT": (362, 236)},
    "rugby": {"LEFT": (67, 316), "RIGHT": (416, 306)},
}

WRONG_WINNERS = [
    {"label": "polo", "side": "LEFT", "wrong_winner": (378, 445), "confirmed_cuff": (31, 154)},
    {"label": "polo", "side": "RIGHT", "wrong_winner": (378, 445), "confirmed_cuff": (439, 142)},
    {"label": "hanging_shirt", "side": "LEFT", "wrong_winner": (365, 241), "confirmed_cuff": (26, 457)},
    {"label": "hanging_shirt", "side": "RIGHT", "wrong_winner": (109, 258), "confirmed_cuff": (442, 447)},
    {"label": "rugby", "side": "LEFT", "wrong_winner": (430, 459), "confirmed_cuff": (51, 488)},
]


def dist(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])


def main():
    for case in WRONG_WINNERS:
        garment = case["label"]
        wrong = case["wrong_winner"]
        underarms = UNDERARMS[garment]

        print(f"\n{'='*70}")
        print(f"{garment} {case['side']} — wrong winner at {wrong}")
        print('='*70)

        for side_name, ua_pt in underarms.items():
            d = dist(wrong, ua_pt)
            print(f"  distance to {side_name}_underarm {ua_pt}: {d:.1f}px")

        # Also check: is it near the CUFF of the opposite side, i.e. 
        # a wraparound of the other side's own true cuff?
        other_side = "RIGHT" if case["side"] == "LEFT" else "LEFT"
        matching_other = next(
            (w for w in WRONG_WINNERS if w["label"] == garment and w["side"] == other_side),
            None
        )
        if matching_other:
            other_confirmed_cuff = matching_other["confirmed_cuff"]
            d = dist(wrong, other_confirmed_cuff)
            print(f"  distance to OTHER side's confirmed cuff {other_confirmed_cuff}: {d:.1f}px")


if __name__ == "__main__":
    main()