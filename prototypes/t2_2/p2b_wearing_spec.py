"""
Try-On v2 T2-2 P2b (2026-11-20): can the renderer follow a WEARING SPEC consistently?

Run from the FitCheckAI folder (venv active):
    python -m prototypes.t2_2.p2b_wearing_spec

Keji's fit principles: the garment keeps its identity (cut, width, length, details, no
underwear from the product photo); how it's worn is a separate, explicit instruction.
Each of P2's 3 outfits is rendered UNTUCKED and FULLY TUCKED, each garment with a written
spec (silhouette, length, rise):
    max   - Try-On Max balanced, 2 steps. Untucked: bottom then top. Tucked: top then bottom.
            Keji's avatar and the synthetic man.
    flux  - FLUX.2 Pro edit, one pass. SYNTHETIC MAN ONLY (privacy rule 2).
Consistency: synth o2 untucked is run twice per model with another seed (max-r, flux-r).

Outputs: prototypes/t2_2/out/p2b/. Safe to re-run. P2b stops at $4.50, T2-2 at $17.
Blind sheet: out/p2b/index.html (key in key.json).
"""

import html
import io
import json
import random
import time

from prototypes.t2_2.p1_modes import _spent, _charge, OUT
from prototypes.t2_2.p2_outfits import (_fashn_run, _person_uri, _garment_uri, _keji_avatar, _wardrobe,
                                        ONE_PASS, SYNTH, P2)
from prototypes.t2_2.p4_built_avatar import _image_bytes

P2B = OUT / "p2b"
WORK = P2B / "work"
T2_CAP_USD = 17.0
P2B_CAP_USD = 4.50
MAX_STEP_USD = 0.15
FLUX_USD = 0.06
SEEDS = {"max": 42, "max-r": 43, "flux": 7, "flux-r": 8}

KEEP = ("Keep the garment's exact cut, width, length, pattern, seams, pockets and logos. "
        "Do not add or remove any details.")
NO_EXTRAS = "Show only this garment: no underwear, boxers, belts or other clothing from the product photo."

SPEC = {
    "o1": {"top": "a regular-fit black crew-neck T-shirt, regular length ending at the hip",
           "bottom": "olive pleated trousers, relaxed straight leg, full length, high-rise, worn at the natural waist"},
    "o2": {"top": "a regular-fit maroon polo shirt with a cream diagonal sash, a crest and a chest logo, regular length",
           "bottom": "light grey wide-leg jeans, full length to the floor, mid-rise, sitting at the hip"},
    "o3": {"top": "a regular-fit green polo shirt with yellow and navy stripes and a small yellow chest logo, regular length",
           "bottom": "indigo washed relaxed straight-leg jeans, full length, mid-rise, sitting at the hip"},
}
TUCK = {"untucked": "The top is worn untucked, hanging over the waistband.",
        "tucked": "The top is fully tucked into the waistband, so the waistband is visible."}


class _Stop(Exception):
    pass


# ---------------------------------------------------------------- spend

def _p2b_spent() -> float:
    try:
        items = json.loads((OUT / "spend.json").read_text())["items"]
        return sum(i["usd"] for i in items if i["what"].startswith("p2b "))
    except Exception:
        return 0.0


def _can_spend(usd: float) -> bool:
    if _spent() + usd > T2_CAP_USD or _p2b_spent() + usd > P2B_CAP_USD:
        print(f"STOP: cap reached (T2-2 ${_spent():.2f} of ${T2_CAP_USD}, P2b ${_p2b_spent():.2f} of ${P2B_CAP_USD}).")
        return False
    return True


# ---------------------------------------------------------------- renderers

def _max(person: bytes, g: dict, prompt: str, seed: int) -> bytes:
    return _fashn_run("tryon-max", {
        "model_image": _person_uri(person), "product_image": _garment_uri(g),
        "resolution": "1k", "generation_mode": "balanced", "seed": seed, "num_images": 1,
        "output_format": "png", "return_base64": True, "prompt": prompt})


def _max_chain(cell: str, person: bytes, top: dict, bottom: dict, oid: str, tuck: str, seed: int) -> bytes:
    spec = SPEC[oid]
    if tuck == "untucked":
        steps = [(bottom, f"{spec['bottom']}. {KEEP} {NO_EXTRAS}"),
                 (top, f"{spec['top']}. {TUCK['untucked']} {KEEP}")]
    else:
        steps = [(top, f"{spec['top']}. {KEEP}"),
                 (bottom, f"{spec['bottom']}. {TUCK['tucked']} {KEEP} {NO_EXTRAS}")]
    cur = person
    for i, (g, prompt) in enumerate(steps):
        step = WORK / f"{cell}_{i}.png"
        if step.exists():
            cur = step.read_bytes()
            continue
        if not _can_spend(MAX_STEP_USD):
            raise _Stop()
        png = _max(cur, g, prompt, seed)
        _charge(f"p2b max {cell} step{i}", MAX_STEP_USD)
        step.write_bytes(png)
        cur = png
    return cur


def _flux(person: bytes, top: dict, bottom: dict, oid: str, tuck: str, seed: int) -> bytes:
    from PIL import Image
    from tryon.vto_renderer import _ensure_fal_key
    _ensure_fal_key()
    import fal_client
    url = fal_client.upload_image(Image.open(io.BytesIO(person)), format="png")
    spec = SPEC[oid]
    prompt = (f"Dress the person in image 1 in exactly these two garments. Image 2 is the top: {spec['top']}. "
              f"Image 3 is the bottom: {spec['bottom']}. {TUCK[tuck]} "
              "Change ONLY the clothes. Keep the person's face, skin colour, hair, body shape, pose, framing "
              "and background exactly the same. Keep each garment's exact cut, width, length, colour, pattern, "
              "seams, pockets and logos; do not add or remove any details. Show no underwear, boxers or belts. "
              "Photorealistic, no text.")
    endpoint, args = ONE_PASS["flux"]
    result = fal_client.subscribe(endpoint, arguments={
        "prompt": prompt, "image_urls": [url, top["clean_image_url"], bottom["clean_image_url"]],
        **{**args, "seed": seed}})
    images = (result or {}).get("images") or []
    if not images or not images[0].get("url"):
        raise RuntimeError("flux: no image returned")
    return _image_bytes(images[0]["url"])


def _make(way: str, row: str, subj: str, person: bytes, top: dict, bottom: dict, oid: str, tuck: str) -> bytes:
    seed = SEEDS[way]
    if way.startswith("max"):
        return _max_chain(f"{row}_{way}", person, top, bottom, oid, tuck, seed)
    if subj != "synth":
        raise RuntimeError("privacy rule: one-pass models only ever see the synthetic person")
    if not _can_spend(FLUX_USD):
        raise _Stop()
    png = _flux(person, top, bottom, oid, tuck, seed)
    _charge(f"p2b {way} {row}", FLUX_USD)
    return png


# ---------------------------------------------------------------- sheet

def _sheet(plan: list, key: dict):
    rows = []
    for row, subj, oid, tuck, top, bottom, ways in plan:
        e = key.get(row)
        if not e:
            continue
        who = "../p2/keji_avatar.png" if subj == "keji" else "../p4b/s1_Y.png"
        spec = SPEC[oid]
        thumbs = (f'<img class="g" src="{who}"><img class="g" src="{html.escape(top["clean_image_url"])}">'
                  f'<img class="g" src="{html.escape(bottom["clean_image_url"])}">')
        cells = [f'<td class="s"><b>{row}</b><br><b>Asked for:</b><br>top: {html.escape(spec["top"])}<br>'
                 f'bottom: {html.escape(spec["bottom"])}<br><b>{html.escape(TUCK[tuck])}</b><br>{thumbs}</td>']
        for label in sorted(e["labels"]):
            f = f"{row}_{label}.png"
            if not (P2B / f).exists():
                cells.append(f"<td>{label}: not made</td>")
                continue
            k = f"{row}|{label}"
            cells.append(
                f'<td><img src="{f}"><div><b>{label}</b></div><div>'
                f'same person <input type="number" min="1" max="5" data-k="{k}|person"><br>'
                f'garment kept <input type="number" min="1" max="5" data-k="{k}|garment"><br>'
                f'tuck as asked <input type="number" min="1" max="5" data-k="{k}|tuck"><br>'
                f'rise as asked <input type="number" min="1" max="5" data-k="{k}|rise"><br>'
                f'realistic <input type="number" min="1" max="5" data-k="{k}|real"></div></td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>T2-2 P2b wearing spec</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px}} td{{vertical-align:top;padding:6px}}
td.s{{width:260px;font-size:14px}} img{{height:380px;border:1px solid #ccc;background:#fafafa}}
img.g{{height:80px;margin:2px}} input{{width:3em}} textarea{{width:100%;height:6em}}</style></head>
<body><h2>T2-2 P2b: did it do what we asked? (1 = no / bad, 5 = exactly / great)</h2>
<p>Left of each row: what the AI was asked for. Letters are shuffled per row.<br>
<b>same person</b> = face, skin, body unchanged. <b>garment kept</b> = cut, width, length, pattern, logos, no added
details, no boxers. <b>tuck as asked</b> and <b>rise as asked</b> = followed the instruction on the left.
<b>realistic</b> = looks like a real photo. Judge against the instruction, not your taste.</p>
<table>{''.join(rows)}</table>
<p><button onclick="go()">Copy my scores</button></p><textarea id="o" readonly></textarea>
<script>function go(){{const s={{}};document.querySelectorAll('input').forEach(i=>{{s[i.dataset.k]=i.value?Number(i.value):null}});
const t=JSON.stringify(s);document.getElementById('o').value=t;navigator.clipboard&&navigator.clipboard.writeText(t)}}</script>
</body></html>"""
    (P2B / "index.html").write_text(page, encoding="utf-8")


# ---------------------------------------------------------------- main

def main():
    P2B.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    p2_key_path = P2 / "key.json"
    if not p2_key_path.exists() or not SYNTH.exists():
        raise SystemExit("P2's key.json and P4b's s1_Y.png are needed - run P2 and P4b first.")
    p2_key = json.loads(p2_key_path.read_text())
    by_id = {r["item_id"]: r for r in _wardrobe()}
    outfits = {}
    for oid in SPEC:
        ids = p2_key.get(f"keji-{oid}", {}).get("garments") or []
        if len(ids) < 2 or ids[0] not in by_id or ids[1] not in by_id:
            raise SystemExit(f"{oid}: garments from P2 not found in the wardrobe.")
        outfits[oid] = (by_id[ids[0]], by_id[ids[1]])
    subjects = {"keji": _keji_avatar(), "synth": SYNTH.read_bytes()}

    plan = []
    for subj in ("keji", "synth"):
        for oid, (top, bottom) in outfits.items():
            for tuck in ("untucked", "tucked"):
                ways = ["max"] if subj == "keji" else ["max", "flux"]
                if subj == "synth" and oid == "o2" and tuck == "untucked":
                    ways += ["max-r", "flux-r"]
                plan.append((f"{subj}-{oid}-{tuck}", subj, oid, tuck, top, bottom, ways))

    key_path = P2B / "key.json"
    key = json.loads(key_path.read_text()) if key_path.exists() else {}
    save = lambda: key_path.write_text(json.dumps(key, indent=2), encoding="utf-8")
    rng = random.Random(202611202)
    try:
        for row, subj, oid, tuck, top, bottom, ways in plan:
            e = key.setdefault(row, {"labels": {}, "timing": {}})
            if not e["labels"]:
                w = list(ways)
                rng.shuffle(w)
                e["labels"] = dict(zip("ABCD", w))
            for label, way in sorted(e["labels"].items()):
                final = P2B / f"{row}_{label}.png"
                if final.exists():
                    continue
                t0 = time.time()
                try:
                    png = _make(way, row, subj, subjects[subj], top, bottom, oid, tuck)
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
    print(f"\nDone. P2b spent ${_p2b_spent():.2f}; T2-2 total ${_spent():.2f}")
    print(f"Open the score sheet: {P2B / 'index.html'}")


if __name__ == "__main__":
    main()
