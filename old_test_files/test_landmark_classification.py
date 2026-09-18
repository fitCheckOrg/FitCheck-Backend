"""
Test classify_landmark against visually-confirmed ground truth
from tonight's investigation.

Run: python test_landmark_classification.py
"""

import sys
sys.path.insert(0, ".")
from workers.wardrobe.landmark_classification import classify_landmark, LandmarkCategory

# Ground truth, from actual visual confirmation earlier tonight
GROUND_TRUTH_CASES = [
    {"label": "polo_baseline A#1 (confirmed real hem corner)",
     "event": {"y_pct": 84.0}, "confidence": 84.0, "reasons": [],
     "expected": LandmarkCategory.HEM_CORNER},

    {"label": "rugby B#1 (confirmed real collar/shoulder point)",
     "event": {"y_pct": 10.2}, "confidence": 89.8, "reasons": [],
     "expected": LandmarkCategory.COLLAR_SHOULDER},

    {"label": "62930780 A#2 post-fix (confirmed real hem corner)",
     "event": {"y_pct": 89.8}, "confidence": 59.8, "reasons": [],
     "expected": LandmarkCategory.HEM_CORNER},

    {"label": "62930780 old B#1 (confirmed underarm re-detection)",
     "event": {"y_pct": 41.0}, "confidence": 50.0,
     "reasons": ["PENALIZED: very close to underarm — likely re-detecting known landmark, not new"],
     "expected": LandmarkCategory.UNDERARM},
]

for case in GROUND_TRUTH_CASES:
    category, explanation = classify_landmark(case["event"], case["confidence"], case["reasons"])
    match = "✅ MATCH" if category == case["expected"] else "❌ MISMATCH"
    print(f"{match}  {case['label']}")
    print(f"    predicted={category.value}  expected={case['expected'].value}")
    print(f"    reason: {explanation}\n")