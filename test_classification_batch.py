"""
Phase B classification — held-out batch test.

Applies classify_landmark to the top event from each of the 14
fresh-validation items, checking specifically whether ambiguous
cases correctly land in AMBIGUOUS rather than being forced into
a category.

Run: python test_classification_batch.py
"""

import sys
sys.path.insert(0, ".")
# Reuses the same pipeline functions from fresh_garment_validation.py — 
# copy that file's imports/functions, or import if refactored into 
# a shared module. For this test, paste the same fetch/score loop 
# from fresh_garment_validation.py, then add after scoring:

from workers.wardrobe.landmark_classification import classify_landmark

# ...after computing `scored` for each item, same as before...
top_score, top_event, top_reasons = scored[0]
category, explanation = classify_landmark(top_event, top_score, top_reasons)
print(f"  Top event classified as: {category.value} — {explanation}")