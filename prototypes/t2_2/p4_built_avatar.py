"""
Try-On v2 T2-2 P4 (2026-11-19): can we build a usable avatar from settings?

Run from the FitCheckAI folder (venv active):
    python -m prototypes.t2_2.p4_built_avatar

For each of 6 setting presets, two text-to-image models on fal each generate a
full-body "model photo" of a SYNTHETIC person (no real photo is sent anywhere -
only the text prompt):
    flux2-pro      fal-ai/flux-2-pro      (Black Forest Labs)  ~$0.045 at 864x1296
    nano-banana-2  fal-ai/nano-banana-2   (Google)             $0.08 at 1K, 2:3
Both are called with sync_mode, so the image isn't kept in fal's request history.

Each generated photo then goes through:
  1. the same pose detection the avatar upload uses (MediaPipe), plus three
     automatic checks: whole body visible, facing the camera, arms away from the body
  2. if it passes, FASHN v1.6 (quality, what the app uses) dresses it in 2 of
     Keji's garments: the pink polo (pattern) and the white jeans

Outputs go to prototypes/t2_2/out/p4/. Safe to re-run (nothing is paid for twice).
Spend is added to the shared T2-2 ledger; P4 stops at P4_CAP_USD and T2-2 at $8.
At the end: out/p4/index.html (blind score sheet: the two models are shuffled
per row as X / Y). The key is out/p4/key.json - don't open it until you've scored.
"""

import base64
import html
import json
import random
import time
from pathlib import Path

from prototypes.t2_2.p1_modes import _spent, _charge, SPEND_CAP_USD, OUT, USER_ID

P4 = OUT / "p4"
P4_CAP_USD = 2.60
FASHN_PRICE = 0.075

MODELS = {
    "flux2-pro": {"endpoint": "fal-ai/flux-2-pro", "usd": 0.045,
                  "args": {"image_size": {"width": 864, "height": 1296}, "output_format": "png",
                           "safety_tolerance": "2", "sync_mode": True, "seed": 7}},
    "nano-banana-2": {"endpoint": "fal-ai/nano-banana-2", "usd": 0.08,
                      "args": {"aspect_ratio": "2:3", "resolution": "1K", "num_images": 1,
                               "output_format": "png", "sync_mode": True, "seed": 7}},
}

PRESETS = [
    {"id": "p1", "gender": "man", "skin": "deep dark brown", "build": "slim", "hair": "a low-cut fade"},
    {"id": "p2", "gender": "woman", "skin": "deep dark brown", "build": "curvy", "hair": "long box braids tied back"},
    {"id": "p3", "gender": "man", "skin": "medium brown", "build": "plus-size, broad", "hair": "a short afro"},
    {"id": "p4", "gender": "woman", "skin": "light brown", "build": "average", "hair": "a natural afro puff"},
    {"id": "p5", "gender": "man", "skin": "light", "build": "athletic", "hair": "short straight hair"},
    {"id": "p6", "gender": "woman", "skin": "medium-deep brown", "build": "plus-size", "hair": "shoulder-length locs"},
]

PROMPT = (
    "Full-body studio photograph of an adult {gender} in their twenties with {skin} skin, "
    "a {build} build and {hair}. Standing straight and facing the camera, arms relaxed and "
    "held slightly away from the body, hands open, feet hip-width apart, neutral expression. "
    "Wearing a plain fitted light grey crew-neck t-shirt and plain fitted light grey shorts, barefoot. "
    "The whole body is visible from the top of the head to the feet, with space above and below. "
    "Plain light grey seamless studio background, soft even lighting, sharp focus, realistic skin "
    "texture, natural proportions, photorealistic, no text, no logos."
)


def _settings_text(p: dict) -> str:
    return f"{p['gender']}, {p['skin']} skin, {p['build']} build, {p['hair']}"


# ---------------------------------------------------------------- generation

def _image_bytes(url: str) -> bytes:
    if url.startswith("data:"):
        return base64.b64decode(url.split(",", 1)[1])
    import requests
    return requests.get(url, timeout=60).content


def _generate(model: str, preset: dict) -> bytes:
    from tryon.vto_renderer import _ensure_fal_key
    _ensure_fal_key()
    import fal_client
    m = MODELS[model]
    result = fal_client.subscribe(m["endpoint"], arguments={"prompt": PROMPT.format(**preset), **m["args"]})
    images = (result or {}).get("images") or []
    if not images or not images[0].get("url"):
        raise RuntimeError(f"{model}: no image returned")
    return _image_bytes(images[0]["url"])


# ---------------------------------------------------------------- checks

NAMES = ["nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
         "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder",
         "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky",
         "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
         "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
         "right_heel", "left_foot_index", "right_foot_index"]


def _points(kp) -> dict:
    """Accept the shapes pose_keypoints may come in: a list of 33, a dict holding that
    list, or a dict keyed by landmark name. Returns {name: {x, y, visibility}}."""
    if isinstance(kp, dict):
        lowered = {str(k).lower(): v for k, v in kp.items()}
        if "left_shoulder" in lowered:
            return {n: lowered[n] for n in NAMES if n in lowered}
        for v in kp.values():
            if isinstance(v, list) and len(v) >= 33:
                kp = v
                break
    if isinstance(kp, list) and len(kp) >= 33:
        return {n: kp[i] for i, n in enumerate(NAMES)}
    return {}


def _checks(img: bytes) -> dict:
    from workers.avatar.pose_estimator import estimate_pose
    out = {"pose": False, "whole_body": False, "facing": False, "arms_away": False}
    try:
        kp = estimate_pose(img)
    except Exception as exc:
        out["note"] = f"pose error: {type(exc).__name__}"
        return out
    pts = _points(kp) if kp else {}
    if not pts:
        if kp:
            out["note"] = f"unknown pose format: {type(kp).__name__} {str(kp)[:120]}"
            print("  (pose came back in a shape this script doesn't read: " + out["note"] + ")")
        return out
    out["pose"] = True

    def p(n):
        v = pts.get(n) or {}
        return float(v.get("x", -1)), float(v.get("y", -1)), float(v.get("visibility", 1.0))

    seen = lambda n: p(n)[2] >= 0.5 and 0.0 <= p(n)[0] <= 1.0 and 0.0 <= p(n)[1] <= 1.0
    out["whole_body"] = all(seen(n) for n in ("nose", "left_ankle", "right_ankle", "left_wrist", "right_wrist"))
    sw = abs(p("left_shoulder")[0] - p("right_shoulder")[0])
    hw = abs(p("left_hip")[0] - p("right_hip")[0])
    out["facing"] = seen("left_shoulder") and seen("right_shoulder") and sw >= 0.12 and hw >= 0.06
    gap_l = abs(p("left_wrist")[0] - p("left_hip")[0])
    gap_r = abs(p("right_wrist")[0] - p("right_hip")[0])
    out["arms_away"] = sw > 0 and min(gap_l, gap_r) >= 0.25 * sw
    return out


def _passed(c: dict) -> bool:
    return all(c.get(k) for k in ("pose", "whole_body", "facing", "arms_away"))


# ---------------------------------------------------------------- try-on

def _garments() -> list[dict]:
    from core.storage.supabase_client import supabase
    rows = supabase.table("closet_items").select("item_id, category, clean_image_url")\
        .eq("user_id", USER_ID).in_("item_id", [
            "2a32b5bb-c3ea-48f9-ac8d-5849eee866ba",      # pink striped polo
            "86cdf563-f862-47e7-af5e-e6efd2447bd7",      # white jeans
        ]).execute().data or []
    return sorted(rows, key=lambda r: r["category"] != "top")


def _dress(person: bytes, garment: dict) -> bytes:
    from tryon.vto_renderer import FalFashnRenderer
    from prototypes.t2_2.p1_modes import _category
    return FalFashnRenderer(mode="quality").render(person, garment["clean_image_url"],
                                                   _category(garment["category"]))


# ---------------------------------------------------------------- sheet

def _sheet(key: dict, garments: list[dict]):
    rows = []
    for preset in PRESETS:
        e = key.get(preset["id"])
        if not e:
            continue
        cells = [f'<td class="s"><b>{preset["id"]}</b><br>{html.escape(_settings_text(preset))}</td>']
        for label in ("X", "Y"):
            base = f"{preset['id']}_{label}"
            if not (P4 / f"{base}.png").exists():
                cells.append(f"<td>{label}: not generated</td>")
                continue
            c = e["checks"].get(label, {})
            marks = " ".join(f"{k} {'ok' if c.get(k) else 'FAIL'}" for k in ("pose", "whole_body", "facing", "arms_away"))
            imgs = f'<img src="{base}.png">'
            for g in garments:
                f = f"{base}_{g['category']}.png"
                if (P4 / f).exists():
                    imgs += f'<img src="{f}">'
            k = f"{preset['id']}|{label}"
            cells.append(
                f'<td><div><b>{label}</b> <small>{marks}</small></div>{imgs}<div>'
                f'real person <input type="number" min="1" max="5" data-k="{k}|real"> '
                f'matches settings <input type="number" min="1" max="5" data-k="{k}|settings"> '
                f'try-on looks right <input type="number" min="1" max="5" data-k="{k}|tryon"></div></td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>T2-2 P4 built avatars</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px}} td{{vertical-align:top;padding:6px}}
td.s{{width:160px}} img{{height:360px;margin-right:4px;border:1px solid #ccc;background:#fafafa}}
input{{width:3em}} textarea{{width:100%;height:6em}}</style></head>
<body><h2>T2-2 P4: built avatars (synthetic people, 1 = bad, 5 = great)</h2>
<p>Each row is one set of settings. X and Y are two different image models, shuffled per row.
For each: the generated person, then the try-ons on them (only if the automatic checks passed).
Score: does it look like a <b>real person</b>, does it <b>match the settings</b>, does the <b>try-on look right</b>.</p>
<table>{''.join(rows)}</table>
<p><button onclick="go()">Copy my scores</button></p><textarea id="o" readonly></textarea>
<script>function go(){{const s={{}};document.querySelectorAll('input').forEach(i=>{{s[i.dataset.k]=i.value?Number(i.value):null}});
const t=JSON.stringify(s);document.getElementById('o').value=t;navigator.clipboard&&navigator.clipboard.writeText(t)}}</script>
</body></html>"""
    (P4 / "index.html").write_text(page, encoding="utf-8")


# ---------------------------------------------------------------- main

def _p4_spent() -> float:
    try:
        items = json.loads((OUT / "spend.json").read_text())["items"]
        return sum(i["usd"] for i in items if i["what"].startswith("p4 "))
    except Exception:
        return 0.0


def _can_spend(usd: float) -> bool:
    if _spent() + usd > SPEND_CAP_USD or _p4_spent() + usd > P4_CAP_USD:
        print(f"STOP: cap reached (T2-2 ${_spent():.2f} of ${SPEND_CAP_USD}, P4 ${_p4_spent():.2f} of ${P4_CAP_USD}).")
        return False
    return True


def main():
    P4.mkdir(parents=True, exist_ok=True)
    key_path = P4 / "key.json"
    key = json.loads(key_path.read_text()) if key_path.exists() else {}
    garments = _garments()
    print(f"garments for the try-ons: {', '.join(g['category'] for g in garments)}")
    rng = random.Random(20261119)
    save = lambda: key_path.write_text(json.dumps(key, indent=2), encoding="utf-8")

    for preset in PRESETS:
        e = key.setdefault(preset["id"], {"settings": _settings_text(preset), "labels": {}, "checks": {}, "timing": {}})
        if not e["labels"]:
            models = list(MODELS)
            rng.shuffle(models)
            e["labels"] = dict(zip(("X", "Y"), models))
        for label, model in sorted(e["labels"].items()):
            base = f"{preset['id']}_{label}"
            photo = P4 / f"{base}.png"
            if not photo.exists():
                if not _can_spend(MODELS[model]["usd"]):
                    _sheet(key, garments); save(); return
                t0 = time.time()
                try:
                    img = _generate(model, preset)
                    _charge(f"p4 gen {model} {preset['id']}", MODELS[model]["usd"])
                    photo.write_bytes(img)
                    e["timing"][model] = round(time.time() - t0, 1)
                except Exception as exc:
                    e["timing"][model] = {"error": str(exc)[:200]}
                    print(f"  {preset['id']} {label}: generation FAILED - {exc}")
                    save()
                    continue
            if label not in e["checks"]:
                e["checks"][label] = _checks(photo.read_bytes())
            c = e["checks"][label]
            print(f"  {preset['id']} {label}: generated, checks "
                  + " ".join(f"{k}={'ok' if c.get(k) else 'FAIL'}" for k in ("pose", "whole_body", "facing", "arms_away")))
            if not _passed(c):
                save()
                continue
            for g in garments:
                dest = P4 / f"{base}_{g['category']}.png"
                if dest.exists():
                    continue
                if not _can_spend(FASHN_PRICE):
                    _sheet(key, garments); save(); return
                try:
                    png = _dress(photo.read_bytes(), g)
                    _charge(f"p4 tryon {model} {preset['id']} {g['category']}", FASHN_PRICE)
                    dest.write_bytes(png)
                    print(f"  {preset['id']} {label}: try-on {g['category']} done")
                except Exception as exc:
                    print(f"  {preset['id']} {label}: try-on {g['category']} FAILED - {exc}")
            save()
    _sheet(key, garments)
    save()
    print(f"\nDone. P4 spent ${_p4_spent():.2f}; T2-2 total ${_spent():.2f}")
    print(f"Open the score sheet: {P4 / 'index.html'}")


if __name__ == "__main__":
    main()
