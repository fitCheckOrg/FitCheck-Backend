"""
Try-On v2 T2-2 P4b (2026-11-19): built avatar, option 3 - dummy -> "Make it real".

Run from the FitCheckAI folder (venv active):
    python -m prototypes.t2_2.p4b_dummy_to_photo

1. Draws 4 grey mannequins in code, each with its own body shape (shoulders, waist,
   hips, height, limb thickness) - this stands in for the slider "dummy" the user shapes.
2. Two image-EDIT models on fal turn each mannequin into a realistic person with the
   chosen skin tone and hair, keeping the mannequin's exact shape and pose:
       flux2-pro-edit    fal-ai/flux-2-pro/edit     (Black Forest Labs)  ~$0.06
       nano-banana-2     fal-ai/nano-banana-2/edit  (Google)             $0.08
   Only the drawn mannequin and the text go to the provider - no real photo (Keji's rule).
   The person is asked to wear fitted BLACK shorts + tank (P4 lesson: grey clothes
   blended into the try-ons).
3. People who pass the avatar checks are dressed, using what won P1:
       top    = black tee,              FASHN v1.6 quality ($0.075)
       bottom = olive pleated trousers, FASHN Try-On Max balanced ($0.15)

Outputs: prototypes/t2_2/out/p4b/. Safe to re-run. P4b stops at $2.40, T2-2 at $8.
Blind sheet: out/p4b/index.html (models shuffled per row as X / Y; key in key.json).
"""

import base64
import html
import io
import json
import random
import time
from pathlib import Path

from prototypes.t2_2.p1_modes import _spent, _charge, SPEND_CAP_USD, OUT, USER_ID, FASHN_API, _fashn_key, _category
from prototypes.t2_2.p4_built_avatar import _checks, _passed, _image_bytes

P4B = OUT / "p4b"
P4B_CAP_USD = 2.40
W, H = 864, 1296

MODELS = {
    "flux2-pro-edit": {"endpoint": "fal-ai/flux-2-pro/edit", "usd": 0.06,
                       "args": {"image_size": {"width": W, "height": H}, "output_format": "png",
                                "safety_tolerance": "2", "sync_mode": True, "seed": 7}},
    "nano-banana-2": {"endpoint": "fal-ai/nano-banana-2/edit", "usd": 0.08,
                      "args": {"aspect_ratio": "2:3", "resolution": "1K", "num_images": 1,
                               "output_format": "png", "sync_mode": True, "seed": 7}},
}

# Body shape = fractions of the canvas. height: share of the canvas the body fills.
SHAPES = [
    {"id": "s1", "gender": "man", "skin": "deep dark brown", "hair": "a low-cut fade",
     "label": "tall, broad shoulders, narrow hips",
     "height": 0.90, "shoulder": 0.30, "waist": 0.17, "hip": 0.19, "limb": 1.05},
    {"id": "s2", "gender": "woman", "skin": "deep dark brown", "hair": "long box braids tied back",
     "label": "average height, curvy: narrow waist, wide hips",
     "height": 0.84, "shoulder": 0.22, "waist": 0.15, "hip": 0.28, "limb": 1.10},
    {"id": "s3", "gender": "man", "skin": "medium brown", "hair": "a short afro",
     "label": "average height, plus-size: wide waist",
     "height": 0.84, "shoulder": 0.29, "waist": 0.29, "hip": 0.28, "limb": 1.35},
    {"id": "s4", "gender": "woman", "skin": "light brown", "hair": "a natural afro puff",
     "label": "short, slim and petite",
     "height": 0.76, "shoulder": 0.19, "waist": 0.13, "hip": 0.18, "limb": 0.80},
]

PROMPT = (
    "Turn this grey mannequin into a photorealistic full-body studio photograph of a real adult "
    "{gender} in their twenties with {skin} skin and {hair}. Keep EXACTLY the same body shape and "
    "proportions as the mannequin ({label}): the same shoulder width, waist, hips, limb thickness and "
    "height, the same pose with the arms held slightly away from the body, and the same framing. "
    "The person wears a fitted plain black tank top and fitted plain black cycling shorts, barefoot. "
    "Plain light grey seamless studio background, soft even lighting, sharp focus, realistic skin "
    "texture, neutral expression, photorealistic, no text, no logos."
)

TOP_ID = "d20577a1-01e9-4d6f-87a1-5ca89c3badba"       # black tee (v1.6 quality won P1)
BOTTOM_ID = "ca98a352-8083-4a8d-b446-9e1afd631cec"    # olive pleated trousers (Max balanced won P1)


# ---------------------------------------------------------------- the dummy

def draw_mannequin(s: dict) -> bytes:
    """A simple front-facing grey mannequin with the shape's proportions
    (limbs drawn as thick rounded segments, arms clear of the body)."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (W, H), (232, 232, 234))
    d = ImageDraw.Draw(img)
    grey = (150, 150, 156)
    cx = W / 2
    body_h = s["height"] * H
    top = (H - body_h) / 2
    u = body_h / 7.5                                   # one "head" unit
    limb = s["limb"]
    sw, ww, hw = s["shoulder"] * W / 2, s["waist"] * W / 2, s["hip"] * W / 2
    y_chin = top + u
    y_sh = y_chin + 0.35 * u
    y_waist = y_sh + 1.5 * u
    y_hip = y_waist + 0.7 * u
    y_crotch = y_hip + 0.45 * u
    y_knee = y_crotch + 1.75 * u
    y_ankle = top + body_h - 0.2 * u

    def seg(p, q, w):
        d.line([p, q], fill=grey, width=int(w))
        r = w / 2
        for x, y in (p, q):
            d.ellipse([x - r, y - r, x + r, y + r], fill=grey)

    # head + neck
    d.ellipse([cx - 0.36 * u, top, cx + 0.36 * u, y_chin], fill=grey)
    seg((cx, y_chin - 0.1 * u), (cx, y_sh + 0.1 * u), 0.26 * u)
    # torso (rounded shoulders)
    d.polygon([(cx - sw + 0.1 * u, y_sh - 0.08 * u), (cx + sw - 0.1 * u, y_sh - 0.08 * u),
               (cx + sw, y_sh + 0.15 * u), (cx + ww, y_waist), (cx + hw, y_hip),
               (cx + hw * 0.9, y_crotch), (cx - hw * 0.9, y_crotch), (cx - hw, y_hip),
               (cx - ww, y_waist), (cx - sw, y_sh + 0.15 * u)], fill=grey)
    # legs: thigh + shin per side, feet
    thigh, calf = 0.36 * u * limb, 0.26 * u * limb
    for side in (-1, 1):
        hip_j = (cx + side * hw * 0.5, y_hip + 0.15 * u)
        knee = (cx + side * max(hw * 0.42, 0.24 * u), y_knee)
        ankle = (cx + side * max(hw * 0.38, 0.22 * u), y_ankle)
        seg(hip_j, knee, thigh)
        seg(knee, ankle, calf)
        d.ellipse([ankle[0] - 0.16 * u, ankle[1] - 0.05 * u, ankle[0] + 0.16 * u, ankle[1] + 0.14 * u], fill=grey)
    # arms: held away from the body
    upper, fore = 0.2 * u * limb, 0.15 * u * limb
    for side in (-1, 1):
        sh_j = (cx + side * (sw - upper * 0.35), y_sh + 0.2 * u)
        elbow = (cx + side * (max(sw, ww) + 0.35 * u + upper * 0.5), y_waist - 0.1 * u)
        wrist = (cx + side * (max(sw, hw) + 0.6 * u + fore * 0.5), y_crotch)
        seg(sh_j, elbow, upper)
        seg(elbow, wrist, fore)
        d.ellipse([wrist[0] - 0.11 * u, wrist[1] - 0.05 * u, wrist[0] + 0.11 * u, wrist[1] + 0.3 * u], fill=grey)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- make it real

def make_real(model: str, s: dict, dummy: bytes) -> bytes:
    from PIL import Image
    from tryon.vto_renderer import _ensure_fal_key
    _ensure_fal_key()
    import fal_client
    url = fal_client.upload_image(Image.open(io.BytesIO(dummy)), format="png")
    m = MODELS[model]
    result = fal_client.subscribe(m["endpoint"], arguments={
        "prompt": PROMPT.format(**s), "image_urls": [url], **m["args"]})
    images = (result or {}).get("images") or []
    if not images or not images[0].get("url"):
        raise RuntimeError(f"{model}: no image returned")
    return _image_bytes(images[0]["url"])


# ---------------------------------------------------------------- try-ons

def _garment(item_id: str) -> dict:
    from core.storage.supabase_client import supabase
    rows = supabase.table("closet_items").select("item_id, category, clean_image_url")\
        .eq("user_id", USER_ID).eq("item_id", item_id).limit(1).execute().data
    if not rows:
        raise SystemExit(f"garment {item_id} not found")
    return rows[0]


def _top_v16(person: bytes, g: dict) -> bytes:
    from tryon.vto_renderer import FalFashnRenderer
    return FalFashnRenderer(mode="quality").render(person, g["clean_image_url"], _category(g["category"]))


def _bottom_max(person: bytes, g: dict) -> bytes:
    import requests
    from tryon.vto_renderer import _to_pil, _shrink, _flatten_on_white, _data_uri, PERSON_MAX_SIDE
    headers = {"Authorization": f"Bearer {_fashn_key()}", "Content-Type": "application/json"}
    body = {"model_name": "tryon-max", "inputs": {
        "model_image": _data_uri(_shrink(_to_pil(person, "person").convert("RGB"), PERSON_MAX_SIDE), "JPEG"),
        "product_image": _data_uri(_flatten_on_white(_to_pil(g["clean_image_url"], "garment")), "PNG"),
        "resolution": "1k", "generation_mode": "balanced", "seed": 42, "num_images": 1,
        "output_format": "png", "return_base64": True}}
    r = requests.post(f"{FASHN_API}/run", json=body, headers=headers, timeout=60)
    if r.status_code != 200 or not r.json().get("id"):
        raise RuntimeError(f"Try-On Max refused ({r.status_code}): {r.text[:200]}")
    pid, t0 = r.json()["id"], time.time()
    while time.time() - t0 < 300:
        time.sleep(2)
        st = requests.get(f"{FASHN_API}/status/{pid}", headers=headers, timeout=30).json()
        if st.get("status") == "completed":
            out = (st.get("output") or [None])[0]
            if not out:
                raise RuntimeError("Try-On Max returned no image")
            return _image_bytes(out) if out.startswith(("data:", "http")) else base64.b64decode(out)
        if st.get("status") in ("failed", "canceled", "time_out"):
            raise RuntimeError(f"Try-On Max {st.get('status')}: {st.get('error')}")
    raise RuntimeError("Try-On Max timed out")


TRYONS = [("top", TOP_ID, _top_v16, 0.075), ("bottom", BOTTOM_ID, _bottom_max, 0.15)]


# ---------------------------------------------------------------- spend

def _p4b_spent() -> float:
    try:
        items = json.loads((OUT / "spend.json").read_text())["items"]
        return sum(i["usd"] for i in items if i["what"].startswith("p4b "))
    except Exception:
        return 0.0


def _can_spend(usd: float) -> bool:
    if _spent() + usd > SPEND_CAP_USD or _p4b_spent() + usd > P4B_CAP_USD:
        print(f"STOP: cap reached (T2-2 ${_spent():.2f} of ${SPEND_CAP_USD}, P4b ${_p4b_spent():.2f} of ${P4B_CAP_USD}).")
        return False
    return True


# ---------------------------------------------------------------- sheet

def _sheet(key: dict):
    rows = []
    for s in SHAPES:
        e = key.get(s["id"])
        if not e:
            continue
        cells = [f'<td class="s"><b>{s["id"]}</b><br>{html.escape(s["gender"] + ", " + s["skin"] + " skin, " + s["hair"])}'
                 f'<br><i>{html.escape(s["label"])}</i><br><img src="{s["id"]}_dummy.png"><div>the dummy</div></td>']
        for label in ("X", "Y"):
            base = f"{s['id']}_{label}"
            if not (P4B / f"{base}.png").exists():
                cells.append(f"<td>{label}: not generated</td>")
                continue
            c = e["checks"].get(label, {})
            marks = " ".join(f"{k} {'ok' if c.get(k) else 'FAIL'}" for k in ("pose", "whole_body", "facing", "arms_away"))
            imgs = f'<img src="{base}.png">' + "".join(
                f'<img src="{base}_{part}.png">' for part, *_ in TRYONS if (P4B / f"{base}_{part}.png").exists())
            k = f"{s['id']}|{label}"
            cells.append(
                f'<td><div><b>{label}</b> <small>{marks}</small></div>{imgs}<div>'
                f'real person <input type="number" min="1" max="5" data-k="{k}|real"> '
                f'same shape as dummy <input type="number" min="1" max="5" data-k="{k}|shape"> '
                f'top try-on <input type="number" min="1" max="5" data-k="{k}|top"> '
                f'trousers try-on <input type="number" min="1" max="5" data-k="{k}|bottom"></div></td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>T2-2 P4b dummy to photo</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px}} td{{vertical-align:top;padding:6px}}
td.s{{width:230px}} img{{height:360px;margin-right:4px;border:1px solid #ccc;background:#fafafa}}
input{{width:3em}} textarea{{width:100%;height:6em}}</style></head>
<body><h2>T2-2 P4b: dummy &rarr; "Make it real" (1 = bad, 5 = great)</h2>
<p>Left: the dummy (its shape is what the user would set with sliders). X and Y: two image models
turning it into a real-looking person (shuffled per row), then the black tee and the olive trousers on them.</p>
<table>{''.join(rows)}</table>
<p><button onclick="go()">Copy my scores</button></p><textarea id="o" readonly></textarea>
<script>function go(){{const s={{}};document.querySelectorAll('input').forEach(i=>{{s[i.dataset.k]=i.value?Number(i.value):null}});
const t=JSON.stringify(s);document.getElementById('o').value=t;navigator.clipboard&&navigator.clipboard.writeText(t)}}</script>
</body></html>"""
    (P4B / "index.html").write_text(page, encoding="utf-8")


# ---------------------------------------------------------------- main

def main():
    P4B.mkdir(parents=True, exist_ok=True)
    if not _fashn_key():
        raise SystemExit("FASHN_API_KEY not found in .env (needed for the trousers).")
    key_path = P4B / "key.json"
    key = json.loads(key_path.read_text()) if key_path.exists() else {}
    save = lambda: key_path.write_text(json.dumps(key, indent=2), encoding="utf-8")
    garments = {gid: _garment(gid) for _, gid, _, _ in TRYONS}
    rng = random.Random(201911192)

    for s in SHAPES:
        dummy_path = P4B / f"{s['id']}_dummy.png"
        if not dummy_path.exists():
            dummy_path.write_bytes(draw_mannequin(s))
        e = key.setdefault(s["id"], {"labels": {}, "checks": {}, "timing": {}})
        if not e["labels"]:
            models = list(MODELS)
            rng.shuffle(models)
            e["labels"] = dict(zip(("X", "Y"), models))
        for label, model in sorted(e["labels"].items()):
            base = f"{s['id']}_{label}"
            photo = P4B / f"{base}.png"
            if not photo.exists():
                if not _can_spend(MODELS[model]["usd"]):
                    _sheet(key); save(); return
                t0 = time.time()
                try:
                    img = make_real(model, s, dummy_path.read_bytes())
                    _charge(f"p4b gen {model} {s['id']}", MODELS[model]["usd"])
                    photo.write_bytes(img)
                    e["timing"][model] = round(time.time() - t0, 1)
                except Exception as exc:
                    print(f"  {s['id']} {label}: make-it-real FAILED - {exc}")
                    save()
                    continue
            if label not in e["checks"]:
                e["checks"][label] = _checks(photo.read_bytes())
            c = e["checks"][label]
            print(f"  {s['id']} {label}: made real, checks "
                  + " ".join(f"{k}={'ok' if c.get(k) else 'FAIL'}" for k in ("pose", "whole_body", "facing", "arms_away")))
            if not _passed(c):
                save()
                continue
            for part, gid, fn, usd in TRYONS:
                dest = P4B / f"{base}_{part}.png"
                if dest.exists():
                    continue
                if not _can_spend(usd):
                    _sheet(key); save(); return
                try:
                    png = fn(photo.read_bytes(), garments[gid])
                    _charge(f"p4b tryon {part} {model} {s['id']}", usd)
                    dest.write_bytes(png)
                    print(f"  {s['id']} {label}: {part} try-on done")
                except Exception as exc:
                    print(f"  {s['id']} {label}: {part} try-on FAILED - {exc}")
            save()
    _sheet(key)
    save()
    print(f"\nDone. P4b spent ${_p4b_spent():.2f}; T2-2 total ${_spent():.2f}")
    print(f"Open the score sheet: {P4B / 'index.html'}")


if __name__ == "__main__":
    main()
