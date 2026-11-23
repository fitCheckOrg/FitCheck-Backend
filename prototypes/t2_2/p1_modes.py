"""
Try-On v2 T2-2 P1 (2026-11-19): which FASHN option should render try-ons?

    v16-quality / v16-balanced / v16-performance   FASHN v1.6 on fal (what the app uses), $0.075
    max-fast-1k                                    FASHN Try-On Max, own API, 1 credit ($0.075)
    max-balanced-1k                                FASHN Try-On Max, own API, 2 credits ($0.15)

Run from the FitCheckAI folder (venv active):
    python -m prototypes.t2_2.p1_modes

Try-On Max needs FASHN_API_KEY in FitCheckAI's .env (never printed). Without it the
two Max columns are skipped and can be added later by re-running.
Privacy (Keji, 2026-11-19): only FASHN receives the avatar photo. Try-On Max gets
it as base64 (FASHN deletes its processing copy when the request ends) and returns
base64 (kept 60 min, not on their CDN); FASHN doesn't train on customer content.

* Renders 5 of your garments x 5 options on your current avatar photo,
  ONE AT A TIME (so the timings are fair). Nothing is written to the database
  or the live render cache - images go to prototypes/t2_2/out/p1/.
* Safe to re-run: an image already on disk is never paid for again, and letters
  already used (and scored) keep their meaning; new options get the free letters.
* Spend is counted in prototypes/t2_2/out/spend.json; the whole of T2-2 stops at
  the cap (SPEND_CAP_USD, $8).
* At the end it writes out/p1/index.html - a blind score sheet (the options are
  shuffled and hidden behind letters). The answer key is in out/p1/key.json.
"""

import html
import json
import random
import sys
import time
from pathlib import Path

USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"      # Keji
VARIANTS = {                                         # name -> USD per image
    "v16-quality": 0.075, "v16-balanced": 0.075, "v16-performance": 0.075,
    "max-fast-1k": 0.075, "max-balanced-1k": 0.15,
}
LABELS = ("A", "B", "C", "D", "E")
FASHN_API = "https://api.fashn.ai/v1"
SPEND_CAP_USD = 8.0
STEP_TIMEOUT = 300                                   # seconds per render
OUT = Path(__file__).resolve().parent / "out"
P1 = OUT / "p1"
SPEND = OUT / "spend.json"


# ---------------------------------------------------------------- spend ledger

def _spent() -> float:
    try:
        return float(json.loads(SPEND.read_text())["total_usd"])
    except Exception:
        return 0.0


def _charge(what: str, usd: float) -> None:
    data = {"total_usd": 0.0, "items": []}
    try:
        data = json.loads(SPEND.read_text())
    except Exception:
        pass
    data["total_usd"] = round(float(data["total_usd"]) + usd, 4)
    data["items"].append({"what": what, "usd": usd, "at": time.strftime("%Y-%m-%d %H:%M:%S")})
    SPEND.parent.mkdir(parents=True, exist_ok=True)
    SPEND.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------- inputs

def _pick_garments():
    from core.storage.supabase_client import supabase
    rows = supabase.table("closet_items").select("item_id, category, clean_image_url, created_at")\
        .eq("user_id", USER_ID).eq("is_archived", False).order("created_at").execute().data or []
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    picks = by_cat.get("top", [])[:2] + by_cat.get("bottom", [])[:2] + by_cat.get("dress", [])[:1]
    for r in by_cat.get("top", [])[2:] + by_cat.get("bottom", [])[2:]:     # fill to 5 if no dress
        if len(picks) >= 5:
            break
        picks.append(r)
    return picks[:5]


def _category(cat: str) -> str:
    from pipelines.tryon_slots import slot_for_category, FASHN_CATEGORY
    return FASHN_CATEGORY[slot_for_category(cat)]


# ---------------------------------------------------------------- FASHN Try-On Max (own API)

def _fashn_key() -> str | None:
    import os
    if os.getenv("FASHN_API_KEY"):
        return os.environ["FASHN_API_KEY"].strip()
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("FASHN_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


def _render_max(person_url: str, garment: dict, mode: str, dest: Path, usd: float) -> dict:
    import base64
    import requests
    from tryon.vto_renderer import _to_pil, _shrink, _flatten_on_white, _data_uri, PERSON_MAX_SIDE
    key = _fashn_key()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    person = _shrink(_to_pil(person_url, "person photo").convert("RGB"), PERSON_MAX_SIDE)
    product = _flatten_on_white(_to_pil(garment["clean_image_url"], "garment photo"))
    body = {"model_name": "tryon-max", "inputs": {
        "model_image": _data_uri(person, "JPEG"), "product_image": _data_uri(product, "PNG"),
        "resolution": "1k", "generation_mode": mode, "seed": 42, "num_images": 1,
        "output_format": "png", "return_base64": True}}
    t0 = time.time()
    r = requests.post(f"{FASHN_API}/run", json=body, headers=headers, timeout=60)
    if r.status_code != 200 or not r.json().get("id"):
        raise RuntimeError(f"Try-On Max refused the request ({r.status_code}): {r.text[:200]}")
    pid = r.json()["id"]
    _charge(f"p1 max-{mode}-1k {garment['item_id']}", usd)
    started = None
    while True:
        time.sleep(2)
        st = requests.get(f"{FASHN_API}/status/{pid}", headers=headers, timeout=30).json()
        status = st.get("status")
        if status == "processing" and started is None:
            started = time.time()
        if status == "completed":
            out = (st.get("output") or [None])[0]
            if not out:
                raise RuntimeError("Try-On Max returned no image")
            if out.startswith("data:"):
                dest.write_bytes(base64.b64decode(out.split(",", 1)[1]))
            elif out.startswith("http"):
                dest.write_bytes(requests.get(out, timeout=60).content)
            else:
                dest.write_bytes(base64.b64decode(out))
            t1 = time.time()
            return {"seconds_total": round(t1 - t0, 1),
                    "seconds_working": round(t1 - (started or t0), 1)}
        if status in ("failed", "canceled", "time_out"):
            raise RuntimeError(f"Try-On Max {status}: {st.get('error')}")
        if time.time() - t0 > STEP_TIMEOUT:
            raise RuntimeError(f"Try-On Max timed out after {STEP_TIMEOUT}s")


# ---------------------------------------------------------------- one render

def _render(person_url: str, garment: dict, variant: str, dest: Path) -> dict:
    usd = VARIANTS[variant]
    if variant.startswith("max-"):
        return _render_max(person_url, garment, variant.split("-")[1], dest, usd)
    from tryon.vto_renderer import FalFashnRenderer
    mode = variant.split("-", 1)[1]
    r = FalFashnRenderer(mode=mode)
    t0 = time.time()
    request_id = r.submit(person_url, garment["clean_image_url"], _category(garment["category"]))
    _charge(f"p1 {variant} {garment['item_id']}", usd)
    started = None
    while True:
        state, value = r.poll(request_id)
        if state == "running" and started is None:
            started = time.time()
        if state == "done":
            dest.write_bytes(r.fetch(value))
            t1 = time.time()
            return {"seconds_total": round(t1 - t0, 1),
                    "seconds_working": round(t1 - (started or t0), 1)}
        if state == "error":
            raise RuntimeError(f"FASHN error ({mode}): {value}")
        if time.time() - t0 > STEP_TIMEOUT:
            r.cancel(request_id)
            raise RuntimeError(f"timed out after {STEP_TIMEOUT}s ({mode})")
        time.sleep(2)


# ---------------------------------------------------------------- score sheet

def _sheet(garments, key):
    rows = []
    for g in garments:
        gid = g["item_id"]
        cells = [f'<td><img src="{html.escape(g["clean_image_url"])}"><div>garment</div></td>']
        for label in sorted(key[gid]["labels"]):
            img = f"{gid}_{label}.png"
            if not (P1 / img).exists():
                continue
            cells.append(
                f'<td><img src="{img}"><div><b>{label}</b> &nbsp;'
                f'looks right <input type="number" min="1" max="5" data-k="{gid}|{label}|look"> &nbsp;'
                f'like my garment <input type="number" min="1" max="5" data-k="{gid}|{label}|garment"></div></td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>T2-2 P1 score sheet</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px}} td{{vertical-align:top;padding:6px;text-align:center}}
img{{height:420px;border:1px solid #ccc;background:#fafafa}} input{{width:3em}} textarea{{width:100%;height:6em}}</style></head>
<body><h2>T2-2 P1: which looks right? (score 1 = bad, 5 = great)</h2>
<p>Each row is one garment rendered several ways (the letters are shuffled per row, and every row's order is different). Score every image, then press the button and paste the text back into the chat.</p>
<table>{''.join(rows)}</table>
<p><button onclick="go()">Copy my scores</button></p><textarea id="o" readonly></textarea>
<script>function go(){{const s={{}};document.querySelectorAll('input').forEach(i=>{{s[i.dataset.k]=i.value?Number(i.value):null}});
const t=JSON.stringify(s);document.getElementById('o').value=t;navigator.clipboard&&navigator.clipboard.writeText(t)}}</script>
</body></html>"""
    (P1 / "index.html").write_text(page, encoding="utf-8")
    (P1 / "key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- main

def main():
    from core.storage.supabase_client import supabase
    P1.mkdir(parents=True, exist_ok=True)
    av = supabase.table("avatars").select("original_photo_url, avatar_version")\
        .eq("user_id", USER_ID).limit(1).execute().data
    if not av:
        sys.exit("No avatar for this user.")
    person = av[0]["original_photo_url"]          # the renderer signs our own S3 links
    garments = _pick_garments()
    if len(garments) < 5:
        print(f"Only {len(garments)} usable garments found - running with those.")
    print(f"avatar v{av[0]['avatar_version']}, garments: " +
          ", ".join(f"{g['category']}" for g in garments))

    key_path = P1 / "key.json"
    key = json.loads(key_path.read_text()) if key_path.exists() else {}
    have_max = bool(_fashn_key())
    if not have_max:
        print("FASHN_API_KEY not found in .env - the Try-On Max columns are skipped for now.")
    rng = random.Random(20261119)
    for g in garments:
        gid = g["item_id"]
        entry = key.setdefault(gid, {"category": g["category"], "labels": {}, "timing": {}})
        # keep letters already used (and scored); older runs named modes without "v16-"
        labels = {l: (v if v in VARIANTS else f"v16-{v}") for l, v in entry["labels"].items()}
        missing = [v for v in VARIANTS if v not in labels.values()]
        rng.shuffle(missing)
        free = [l for l in LABELS if l not in labels]
        labels.update(zip(free, missing))
        entry["labels"] = labels
        entry["timing"] = {(k if k in VARIANTS else f"v16-{k}"): t for k, t in entry["timing"].items()}
        for label, variant in sorted(entry["labels"].items()):
            dest = P1 / f"{gid}_{label}.png"
            if dest.exists():
                print(f"  {g['category']:<7} {label}: already done")
                continue
            if variant.startswith("max-") and not have_max:
                continue
            if _spent() + VARIANTS[variant] > SPEND_CAP_USD:
                print(f"STOP: the T2-2 spend cap (${SPEND_CAP_USD}) would be passed. Spent ${_spent():.2f}.")
                _sheet(garments, key)
                return
            try:
                t = _render(person, g, variant, dest)
                entry["timing"][variant] = t
                print(f"  {g['category']:<7} {label}: done")
            except Exception as exc:
                entry["timing"][variant] = {"error": str(exc)[:200]}
                print(f"  {g['category']:<7} {label}: FAILED - {exc}")
            key_path.write_text(json.dumps(key, indent=2), encoding="utf-8")
    _sheet(garments, key)
    print(f"\nDone. Spent so far in T2-2: ${_spent():.2f}")
    print(f"Open the score sheet: {P1 / 'index.html'}")


if __name__ == "__main__":
    main()
