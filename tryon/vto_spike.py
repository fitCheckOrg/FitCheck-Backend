"""
VTO spike: render the test user's avatar wearing the test garment with a
learned try-on model (FASHN v1.6 on fal.ai). Research code, not production.

Needs:
  - FastAPI server running on :8000 (we read the avatar + garment through it)
  - FAL_KEY in .env (this script loads .env itself)
  - pip install fal-client

Run from the FitCheckAI root:
  python -m tryon.vto_spike

Output: outputs/vto_spike/  (inputs, each result, and a side-by-side compare.png)
Cost: num_samples outputs per run (2 by default).
"""

import io
import os
import sys
from pathlib import Path

import requests
from PIL import Image

TEST_USER_ID = "f4b5f129-db34-40e5-8423-d1b256a15578"
TEST_GARMENT_ID = "a852b031-a407-49db-9df6-7e22ecac2ea6"
API = "http://127.0.0.1:8000/api"
OUT_DIR = Path("outputs/vto_spike")

FASHN_ENDPOINT = "fal-ai/fashn/tryon/v1.6"
NUM_SAMPLES = 2


def load_env(path=".env"):
    """Put .env values into os.environ (without overwriting existing ones)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_json(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    body = r.json()
    return body.get("data", body)


def find_original_photo_url(avatar):
    """Find the original (unprocessed) photo URL without guessing one exact name."""
    candidates = [k for k, v in avatar.items()
                  if isinstance(v, str) and v.startswith("http") and "original" in k.lower()]
    if candidates:
        return avatar[candidates[0]], candidates[0]
    print("\nCould not find an 'original' photo URL in /api/avatar/me.")
    print("Fields returned:")
    for k, v in avatar.items():
        print(f"  {k}: {str(v)[:90]}")
    print("\nPaste this list back to the backend chat.")
    sys.exit(1)


def download(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content))


def flatten_on_white(img):
    """Garment PNGs are transparent; give the model a clean white background."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        return bg
    return img.convert("RGB")


def to_data_uri(path, mime):
    import base64
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def side_by_side(images, height=900):
    resized = [im.convert("RGB").resize((int(im.width * height / im.height), height)) for im in images]
    canvas = Image.new("RGB", (sum(im.width for im in resized), height), (255, 255, 255))
    x = 0
    for im in resized:
        canvas.paste(im, (x, 0))
        x += im.width
    return canvas


def main():
    load_env()
    if not os.environ.get("FAL_KEY"):
        print("FAL_KEY not found. Check the line in .env is exactly FAL_KEY=<your key>.")
        sys.exit(1)

    import fal_client  # imported after FAL_KEY is set

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("1/4 Reading avatar and garment from the backend...")
    avatar = get_json(f"{API}/avatar/me?user_id={TEST_USER_ID}")
    person_url, field = find_original_photo_url(avatar)
    print(f"    avatar photo field: {field}")
    garment = get_json(f"{API}/wardrobe/{TEST_GARMENT_ID}?user_id={TEST_USER_ID}")
    garment_url = garment["clean_image_url"]

    print("2/4 Downloading and preparing inputs...")
    person = download(person_url).convert("RGB")
    garment_img = flatten_on_white(download(garment_url))
    person_path = OUT_DIR / "input_person.jpg"
    garment_path = OUT_DIR / "input_garment.png"
    person.save(person_path, quality=95)
    garment_img.save(garment_path)

    print("3/4 Encoding inputs (sent inline, no fal storage needed)...")
    person_fal = to_data_uri(person_path, "image/jpeg")
    garment_fal = to_data_uri(garment_path, "image/png")

    print(f"4/4 Running FASHN v1.6 ({NUM_SAMPLES} samples, quality mode)... this can take a minute")
    result = fal_client.subscribe(
        FASHN_ENDPOINT,
        arguments={
            "model_image": person_fal,
            "garment_image": garment_fal,
            "category": "tops",
            "garment_photo_type": "flat-lay",
            "mode": "quality",
            "num_samples": NUM_SAMPLES,
            "output_format": "png",
        },
    )

    outputs = []
    for i, img in enumerate(result.get("images", [])):
        out = download(img["url"])
        path = OUT_DIR / f"fashn_{i}.png"
        out.save(path)
        outputs.append(out)
        print(f"    saved {path}")

    if not outputs:
        print("No images returned. Raw result:")
        print(result)
        sys.exit(1)

    compare = side_by_side([person, garment_img] + outputs)
    compare_path = OUT_DIR / "compare.png"
    compare.save(compare_path)
    print(f"\nDone. Open {compare_path}  (original | garment | results)")


if __name__ == "__main__":
    main()