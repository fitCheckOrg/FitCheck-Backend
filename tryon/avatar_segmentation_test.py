"""
Step 3 of the reset architecture — pure investigation, no fitting,
no deformation. Answers one question: can we obtain real body-part
segmentation for the golden avatar using existing technology
(MediaPipe's multiclass selfie segmenter), before building anything
custom?

Reports honestly what categories are actually produced. Does NOT
assume torso/arm/leg separation — MediaPipe's multiclass model
gives skin/clothes/hair/background, which is a different but real
kind of structure. We inspect what we actually get.

One-time setup: downloads the multiclass segmentation model file
(~4MB) if not already present.

Run: python -m tryon.avatar_segmentation_test
"""

import io
import os
import requests
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
OUTPUT_DIR = "outputs"
MODEL_PATH = "selfie_multiclass_256x256.tflite"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"

# Category labels for MediaPipe's multiclass selfie segmenter
CATEGORY_LABELS = ["background", "hair", "body-skin", "face-skin", "clothes", "others"]
CATEGORY_COLORS = [
    (0, 0, 0),        # background
    (139, 69, 19),    # hair - brown
    (255, 200, 150),  # body-skin - tan
    (255, 220, 180),  # face-skin - lighter tan
    (0, 150, 255),    # clothes - blue
    (128, 128, 128),  # others - gray
]


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading segmentation model to {MODEL_PATH}...")
        r = requests.get(MODEL_URL)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("Model downloaded.")
    else:
        print(f"Model already present at {MODEL_PATH}.")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ensure_model()

    print("\nFetching golden avatar...")
    avatar_resp = requests.get(
        f"http://127.0.0.1:8000/api/avatar/me?user_id={AVATAR_USER_ID}"
    ).json()["data"]
    print(f"avatar_version: {avatar_resp.get('avatar_version')}")

    avatar_bytes = requests.get(avatar_resp["processed_photo_url"]).content
    pil_image = Image.open(io.BytesIO(avatar_bytes)).convert("RGB")
    np_image = np.array(pil_image)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image)

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.ImageSegmenterOptions(
        base_options=base_options,
        output_category_mask=True,
    )

    print("\nRunning segmentation...")
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        result = segmenter.segment(mp_image)
        category_mask = result.category_mask.numpy_view()

    unique_categories = np.unique(category_mask)
    print(f"\nCategories found in this image: {unique_categories.tolist()}")
    for cat_id in unique_categories:
        label = CATEGORY_LABELS[cat_id] if cat_id < len(CATEGORY_LABELS) else f"unknown({cat_id})"
        pixel_count = np.sum(category_mask == cat_id)
        pct = pixel_count / category_mask.size * 100
        print(f"  {cat_id} ({label}): {pixel_count} px ({pct:.1f}%)")

    # Build a colored visualization
    category_mask = np.squeeze(category_mask)
    h, w = category_mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cat_id in unique_categories:
        color = CATEGORY_COLORS[cat_id] if cat_id < len(CATEGORY_COLORS) else (255, 0, 255)
        colored[category_mask == cat_id] = color

    # Side-by-side: original | segmentation
    original_resized = np.array(pil_image.resize((w, h)))
    combined = np.concatenate([original_resized, colored], axis=1)
    combined_img = Image.fromarray(combined)
    combined_img.save(f"{OUTPUT_DIR}/avatar_segmentation_test.png")

    print(f"\nSaved: {OUTPUT_DIR}/avatar_segmentation_test.png")
    print(f"\n>>> HONEST CHECK: does this give torso/arm/leg separation, or only "
          f"skin/clothes/hair/background? Inspect the image directly — this "
          f"determines whether this model is sufficient or whether we need a "
          f"different segmentation approach (e.g., a body-part parsing model "
          f"like DensePose or a human-parsing dataset-trained model).")


if __name__ == "__main__":
    main()