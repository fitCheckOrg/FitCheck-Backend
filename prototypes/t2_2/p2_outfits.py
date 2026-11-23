"""
Try-On v2 T2-2 P2 (2026-11-20): whole outfits - top + bottom, and one with a jacket.

Run from the FitCheckAI folder (venv active):
    python -m prototypes.t2_2.p2_outfits

Subjects:
    keji  - Keji's avatar photo. FASHN ONLY (FASHN's own API: v1.6 and Try-On Max).
    synth - the P4b Nano Banana 2 man (out/p4b/s1_Y.png). Not a real person, so the
            one-pass models (Nano Banana 2 edit, FLUX.2 Pro edit) may see him.
Ways of dressing:
    max   - every garment on Try-On Max balanced (tops/jacket: "keep the original length and fit")
    mix   - P1 winners: top + jacket on v1.6 quality, bottom on Try-On Max balanced
    nano  - Nano Banana 2 edit, all garments in one go (synth only)
    flux  - FLUX.2 Pro edit, all garments in one go (synth only)
Plus: the black tee alone on Keji's avatar, Try-On Max fast and balanced, with the same instruction.

2026-11-20: fal removed FASHN v1.6, so v1.6 now runs on FASHN's own API
(model_name "tryon-v1.6", 1 credit = $0.075).

Outputs: prototypes/t2_2/out/p2/. Safe to re-run (every finished step is kept).
P2 stops at $6, T2-2 at $15. Blind sheet: out/p2/index.html (key in key.json).
"""

import html
import io
import json
import random
import time
from pathlib import Path

from prototypes.t2_2.p1_modes import _spent, _charge, OUT, USER_ID, FASHN_API, _fashn_key, _category
from prototypes.t2_2.p4_built_avatar import _image_bytes

P2 = OUT / "p2"
WORK = P2 / "work"
T2_CAP_USD = 15.0
P2_CAP_USD = 6.0
SYNTH = OUT / "p4b" / "s1_Y.png"

TEE = "d20577a1-01e9-4d6f-87a1-5ca89c3badba"           # black tee
TROUSERS = "ca98a352-8083-4a8d-b446-9e1afd631cec"      # olive pleated trousers
JEANS = "86cdf563-f862-47e7-af5e-e6efd2447bd7"         # white jeans
THIRD_BOTTOM = "70c9bb98-0da6-4630-8176-471de4574751"
SKIP_TOPS = {TEE, "f2944c50-5757-4abc-81a6-c1380f8419a0",   # tee + its duplicate
             "2a32b5bb-c3ea-48f9-ac8d-5849eee866ba"}        # polo (back-hem photo)
JACKET_WORDS = ("jacket", "coat", "blazer", "cardigan", "hoodie", "bomber", "overshirt",
                "shacket", "windbreaker", "parka", "gilet", "kimono")
MAX_TOP_PROMPT = "keep the original length and fit of the garment"
USD = {"max": 0.15, "max-fast": 0.075, "v16": 0.075, "nano": 0.08, "flux": 0.06}

ONE_PASS = {
    "nano": ("fal-ai/nano-banana-2/edit", {"aspect_ratio": "2:3", "resolution": "1K", "num_images": 1,
                                          "output_format": "png", "sync_mode": True, "seed": 7}),
    "flux": ("fal-ai/flux-2-pro/edit", {"image_size": {"width": 864, "height": 1296}, "output_format": "png",
                                        "safety_tolerance": "2", "sync_mode": True, "seed": 7}),
}


class _Stop(Exception):
    pass


# ---------------------------------------------------------------- spend

def _p2_spent() -> float:
    try:
        items = json.loads((OUT / "spend.json").read_text())["items"]
        return sum(i["usd"] for i in items if i["what"].startswith("p2 "))
    except Exception:
        return 0.0


def _can_spend(usd: float) -> bool:
    if _spent() + usd > T2_CAP_USD or _p2_spent() + usd > P2_CAP_USD:
        print(f"STOP: cap reached (T2-2 ${_spent():.2f} of ${T2_CAP_USD}, P2 ${_p2_spent():.2f} of ${P2_CAP_USD}).")
        return False
    return True


# ---------------------------------------------------------------- inputs

def _png(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _keji_avatar() -> bytes:
    p = P2 / "keji_avatar.png"
    if p.exists():
        return p.read_bytes()
    from core.storage.supabase_client import supabase
    from tryon.vto_renderer import _to_pil
    av = supabase.table("avatars").select("original_photo_url").eq("user_id", USER_ID).limit(1).execute().data
    if not av:
        raise SystemExit("No avatar for this user.")
    png = _png(_to_pil(av[0]["original_photo_url"], "person photo").convert("RGB"))
    p.write_bytes(png)
    return png


def _wardrobe() -> list:
    from core.storage.supabase_client import supabase
    return supabase.table("closet_items")\
        .select("item_id, category, item_type, dominant_color, pattern, clean_image_url, created_at")\
        .eq("user_id", USER_ID).eq("is_archived", False).order("created_at").execute().data or []


def _is_jacket(r: dict) -> bool:
    t = (r.get("item_type") or "").lower()
    return any(w in t for w in JACKET_WORDS)


def _outfits(rows: list) -> list:
    by_id = {r["item_id"]: r for r in rows}
    for gid in (TEE, TROUSERS, JEANS):
        if gid not in by_id:
            raise SystemExit(f"garment {gid} not found (archived?)")
    tops = [r for r in rows if r["category"] == "top" and r["item_id"] not in SKIP_TOPS
            and not _is_jacket(r) and r.get("clean_image_url")]
    if len(tops) < 2:
        raise SystemExit("Need at least 2 other tops in the wardrobe.")
    colourful = [r for r in tops if (r.get("dominant_color") or "").lower() not in ("black", "")]
    if len(colourful) < 2:
        colourful = tops
    jackets = [r for r in rows if r["category"] == "top" and _is_jacket(r) and r.get("clean_image_url")]
    third = by_id.get(THIRD_BOTTOM, by_id[TROUSERS])
    outfits = [("o1", [by_id[TEE], by_id[TROUSERS]]),
               ("o2", [colourful[0], by_id[JEANS]]),
               ("o3", [colourful[1], third])]
    if jackets:
        outfits.append(("o4", [by_id[TEE], by_id[JEANS], jackets[0]]))
    else:
        print("No jacket/coat/hoodie found among your tops - the jacket outfit is skipped.")
    return outfits


def _describe(g: dict) -> str:
    return " ".join(x for x in (g.get("dominant_color"), g.get("pattern"), g.get("item_type") or g["category"]) if x)


# ---------------------------------------------------------------- FASHN's own API

def _fashn_run(model_name: str, inputs: dict) -> bytes:
    import base64
    import requests
    headers = {"Authorization": f"Bearer {_fashn_key()}", "Content-Type": "application/json"}
    r = requests.post(f"{FASHN_API}/run", json={"model_name": model_name, "inputs": inputs},
                      headers=headers, timeout=60)
    if r.status_code != 200 or not r.json().get("id"):
        raise RuntimeError(f"{model_name} refused ({r.status_code}): {r.text[:200]}")
    pid, t0 = r.json()["id"], time.time()
    while time.time() - t0 < 300:
        time.sleep(2)
        st = requests.get(f"{FASHN_API}/status/{pid}", headers=headers, timeout=30).json()
        if st.get("status") == "completed":
            out = (st.get("output") or [None])[0]
            if not out:
                raise RuntimeError(f"{model_name} returned no image")
            return _image_bytes(out) if out.startswith(("data:", "http")) else base64.b64decode(out)
        if st.get("status") in ("failed", "canceled", "time_out"):
            raise RuntimeError(f"{model_name} {st.get('status')}: {st.get('error')}")
    raise RuntimeError(f"{model_name} timed out")


def _person_uri(person: bytes) -> str:
    from tryon.vto_renderer import _to_pil, _shrink, _data_uri, PERSON_MAX_SIDE
    return _data_uri(_shrink(_to_pil(person, "person").convert("RGB"), PERSON_MAX_SIDE), "JPEG")


def _garment_uri(g: dict) -> str:
    from tryon.vto_renderer import _to_pil, _flatten_on_white, _data_uri
    return _data_uri(_flatten_on_white(_to_pil(g["clean_image_url"], "garment")), "PNG")


def _v16(person: bytes, g: dict) -> bytes:
    return _fashn_run("tryon-v1.6", {
        "model_image": _person_uri(person), "garment_image": _garment_uri(g),
        "category": _category(g["category"]), "mode": "quality", "garment_photo_type": "auto",
        "seed": 42, "num_samples": 1, "output_format": "png", "return_base64": True})


def _max(person: bytes, g: dict, mode: str, prompt) -> bytes:
    inputs = {"model_image": _person_uri(person), "product_image": _garment_uri(g),
              "resolution": "1k", "generation_mode": mode, "seed": 42, "num_images": 1,
              "output_format": "png", "return_base64": True}
    if prompt:
        inputs["prompt"] = prompt
    return _fashn_run("tryon-max", inputs)


# ---------------------------------------------------------------- one-pass (synthetic person only)

def _one_pass(way: str, person: bytes, garments: list) -> bytes:
    from PIL import Image
    from tryon.vto_renderer import _ensure_fal_key
    _ensure_fal_key()
    import fal_client
    url = fal_client.upload_image(Image.open(io.BytesIO(person)), format="png")
    names = ["top", "bottom", "jacket, worn open over the top"]
    parts = [f"image {i + 2} is the {names[i]}" for i in range(len(garments))]
    prompt = ("Dress the person in image 1 in these exact garments: " + "; ".join(parts) + ". "
              "Change ONLY the clothes. Keep the person's face, skin colour, hair, body shape, pose, "
              "framing and background exactly the same. Keep each garment's exact colour, pattern, "
              "logos, length and fit. Photorealistic, no text.")
    endpoint, args = ONE_PASS[way]
    result = fal_client.subscribe(endpoint, arguments={
        "prompt": prompt, "image_urls": [url] + [g["clean_image_url"] for g in garments], **args})
    images = (result or {}).get("images") or []
    if not images or not images[0].get("url"):
        raise RuntimeError(f"{way}: no image returned")
    return _image_bytes(images[0]["url"])


# ---------------------------------------------------------------- one cell

def _step_cost(way: str, i: int) -> float:
    if way == "max" or i == 1:
        return USD["max"]
    return USD["v16"]


def _chain(subj: str, oid: str, way: str, person: bytes, garments: list) -> bytes:
    cur = person
    for i, g in enumerate(garments):
        step = WORK / f"{subj}_{oid}_{way}_{i}.png"
        if step.exists():
            cur = step.read_bytes()
            continue
        usd = _step_cost(way, i)
        if not _can_spend(usd):
            raise _Stop()
        if way == "max" or i == 1:
            png = _max(cur, g, "balanced", None if i == 1 else MAX_TOP_PROMPT)
        else:
            png = _v16(cur, g)
        _charge(f"p2 {way} {subj} {oid} step{i}", usd)
        step.write_bytes(png)
        cur = png
    return cur


def _make(way: str, subj: str, oid: str, person: bytes, garments: list) -> bytes:
    if way in ("max", "mix"):
        return _chain(subj, oid, way, person, garments)
    if way in ONE_PASS:
        if subj != "synth":
            raise RuntimeError("privacy rule: one-pass models only ever see the synthetic person")
        if not _can_spend(USD[way]):
            raise _Stop()
        png = _one_pass(way, person, garments)
        _charge(f"p2 {way} {subj} {oid}", USD[way])
        return png
    if way in ("tee-fast", "tee-balanced"):
        mode = way.split("-", 1)[1]
        usd = USD["max-fast"] if mode == "fast" else USD["max"]
        if not _can_spend(usd):
            raise _Stop()
        png = _max(person, garments[0], mode, MAX_TOP_PROMPT)
        _charge(f"p2 {way} {subj}", usd)
        return png
    raise RuntimeError(f"unknown way {way}")


# ---------------------------------------------------------------- sheet

def _sheet(plan: list, key: dict):
    rows = []
    for row, subj, oid, gs, ways in plan:
        e = key.get(row)
        if not e:
            continue
        who = "keji_avatar.png" if subj == "keji" else "../p4b/s1_Y.png"
        thumbs = f'<img class="g" src="{who}">' + "".join(
            f'<img class="g" src="{html.escape(g["clean_image_url"])}">' for g in gs)
        desc = html.escape(" + ".join(_describe(g) for g in gs))
        cells = [f'<td class="s"><b>{row}</b><br>{desc}<br>{thumbs}</td>']
        for label in sorted(e["labels"]):
            f = f"{row}_{label}.png"
            if not (P2 / f).exists():
                cells.append(f"<td>{label}: not made</td>")
                continue
            k = f"{row}|{label}"
            cells.append(
                f'<td><img src="{f}"><div><b>{label}</b></div><div>'
                f'same person <input type="number" min="1" max="5" data-k="{k}|person"><br>'
                f'garments right <input type="number" min="1" max="5" data-k="{k}|garments"><br>'
                f'fit believable <input type="number" min="1" max="5" data-k="{k}|fit"><br>'
                f'no broken bits <input type="number" min="1" max="5" data-k="{k}|clean"></div></td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>T2-2 P2 whole outfits</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px}} td{{vertical-align:top;padding:6px}}
td.s{{width:230px}} img{{height:380px;border:1px solid #ccc;background:#fafafa}}
img.g{{height:90px;margin:2px}} input{{width:3em}} textarea{{width:100%;height:6em}}</style></head>
<body><h2>T2-2 P2: whole outfits (1 = bad, 5 = great)</h2>
<p>Left: the person and the garments. Letters are shuffled per row.<br>
<b>same person</b> = face, skin colour and body unchanged (your hard rule: anything below 4 means it changed the person).
<b>garments right</b> = colour, pattern, logo, length. <b>fit believable</b> = looks like it's really worn.
<b>no broken bits</b> = hands, edges, extra layers.</p>
<table>{''.join(rows)}</table>
<p><button onclick="go()">Copy my scores</button></p><textarea id="o" readonly></textarea>
<script>function go(){{const s={{}};document.querySelectorAll('input').forEach(i=>{{s[i.dataset.k]=i.value?Number(i.value):null}});
const t=JSON.stringify(s);document.getElementById('o').value=t;navigator.clipboard&&navigator.clipboard.writeText(t)}}</script>
</body></html>"""
    (P2 / "index.html").write_text(page, encoding="utf-8")


# ---------------------------------------------------------------- main

def main():
    P2.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    if not _fashn_key():
        raise SystemExit("FASHN_API_KEY not found in .env.")
    if not SYNTH.exists():
        raise SystemExit("out/p4b/s1_Y.png not found - P2 needs P4b's synthetic man.")
    rows = _wardrobe()
    by_id = {r["item_id"]: r for r in rows}
    outfits = _outfits(rows)
    print("Outfits:")
    for oid, gs in outfits:
        print(f"  {oid}: " + " + ".join(_describe(g) for g in gs))
    subjects = {"keji": (_keji_avatar(), ["max", "mix"]),
                "synth": (SYNTH.read_bytes(), ["max", "mix", "nano", "flux"])}
    plan = [(f"{s}-{oid}", s, oid, gs, ways) for s, (_, ways) in subjects.items() for oid, gs in outfits]
    plan.append(("keji-tee", "keji", "tee", [by_id[TEE]], ["tee-fast", "tee-balanced"]))

    key_path = P2 / "key.json"
    key = json.loads(key_path.read_text()) if key_path.exists() else {}
    save = lambda: key_path.write_text(json.dumps(key, indent=2), encoding="utf-8")
    rng = random.Random(20261120)
    try:
        for row, subj, oid, gs, ways in plan:
            person = subjects[subj][0]
            e = key.setdefault(row, {"labels": {}, "timing": {}, "garments": [g["item_id"] for g in gs]})
            if not e["labels"]:
                w = list(ways)
                rng.shuffle(w)
                e["labels"] = dict(zip("ABCD", w))
            for label, way in sorted(e["labels"].items()):
                final = P2 / f"{row}_{label}.png"
                if final.exists():
                    continue
                t0 = time.time()
                try:
                    png = _make(way, subj, oid, person, gs)
                except _Stop:
                    raise
                except Exception as exc:
                    print(f"  {row} {label}: FAILED - {exc}")
                    save()
                    continue
                final.write_bytes(png)
                e["timing"][way] = round(time.time() - t0, 1)
                print(f"  {row} {label}: done")
                save()
    except _Stop:
        pass
    _sheet(plan, key)
    save()
    print(f"\nDone. P2 spent ${_p2_spent():.2f}; T2-2 total ${_spent():.2f}")
    print(f"Open the score sheet: {P2 / 'index.html'}")


if __name__ == "__main__":
    main()
