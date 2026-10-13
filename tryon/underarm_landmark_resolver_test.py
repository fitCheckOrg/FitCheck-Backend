"""
tryon/underarm_landmark_resolver_test.py

Tests for tryon.underarm_landmark_resolver. SYNTHETIC figures only: they check the resolver's LOGIC (policy, plausibility
rules, return model, never-raises), not how it behaves on real avatars.

Run:  python -m tryon.underarm_landmark_resolver_test        (or:  pytest tryon/underarm_landmark_resolver_test.py)
"""

import json
import numpy as np
from PIL import Image

from tryon.underarm_landmark_resolver import (
    LANDMARK_REVISION, LandmarkSource, ResolutionMethod, resolve_underarm_landmarks,
)

W, H = 223, 420
LO, HI = 62, 161                      # torso columns
Y_SH = 160
LEFT_SH, RIGHT_SH = (40.0, 162.0), (186.0, 160.0)      # span ~146 px, centre x 113
APEX_ROW = 223                        # first row where both arms leave the torso


def figure(apex_l=APEX_ROW, apex_r=APEX_ROW, alpha=True, crease_rows=None, as_pil=True):
    """Head+neck, torso LO..HI from Y_SH to 330, an arm on each side attached until its apex row, then opening."""
    fg = np.zeros((H, W), bool)
    yy, xx = np.mgrid[0:H, 0:W]
    fg |= ((xx - 111) ** 2 + (yy - 115) ** 2) <= 20 ** 2
    fg[100:Y_SH + 1, 105:118] = True
    for y in range(Y_SH, 330):
        fg[y, LO:HI + 1] = True
        for side, apex in (("L", apex_l), ("R", apex_r)):
            gap = 0 if (apex is None or y < apex) else 2 + int((y - apex) * 0.25)
            if side == "L":
                a1 = LO - gap
                a0 = a1 - 20
            else:
                a0 = HI + gap
                a1 = a0 + 20
            fg[y, max(a0, 0):min(a1, W - 1) + 1] = True
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[fg] = (150, 110, 90)
    if crease_rows is not None:                       # dark-skin crease (sum 24) between arm and chest, 2 px wide
        for y in range(*crease_rows):
            rgb[y, LO - 1:LO + 1] = (8, 8, 8)
            rgb[y, HI:HI + 2] = (8, 8, 8)
    a = (fg * 255).astype(np.uint8) if alpha else np.full((H, W), 255, np.uint8)
    arr = np.dstack([rgb, a])
    return Image.fromarray(arr, "RGBA") if as_pil else arr


def resolve(img, dl=(61.0, 229.0), dr=(162.0, 213.0), ls=LEFT_SH, rs=RIGHT_SH):
    return resolve_underarm_landmarks(img, ls, rs, dl, dr)


# ---------------------------------------------------------------------------------------------- the locked policy
def test_plausible_detector_is_kept_even_when_apex_disagrees():
    r = resolve(figure())
    assert r.method is ResolutionMethod.DETECTOR
    assert (r.left.source, r.right.source) == (LandmarkSource.DETECTOR, LandmarkSource.DETECTOR)
    assert (r.left.x, r.left.y) == (61.0, 229.0) and (r.right.x, r.right.y) == (162.0, 213.0)
    # the apex is still computed and recorded, so the disagreement is measurable
    assert abs(r.left_evidence.apex.y - APEX_ROW) <= 1 and abs(r.right_evidence.apex.y - APEX_ROW) <= 1
    assert abs(r.left_evidence.disagreement_y_px - 6.0) <= 1.0
    assert abs(r.right_evidence.disagreement_y_px + 10.0) <= 1.0


def test_missing_detector_side_falls_back_to_apex_giving_mixed():
    r = resolve(figure(), dl=None)
    assert r.method is ResolutionMethod.MIXED
    assert r.left.source is LandmarkSource.SILHOUETTE_APEX and abs(r.left.y - APEX_ROW) <= 1
    assert r.right.source is LandmarkSource.DETECTOR and r.right.y == 213.0


def test_no_detector_at_all_gives_apex_apex():
    r = resolve(figure(), dl=None, dr=None)
    assert r.method is ResolutionMethod.SILHOUETTE_APEX
    assert r.resolved and abs(r.left.y - APEX_ROW) <= 1 and abs(r.right.y - APEX_ROW) <= 1
    assert r.points is not None


def test_implausible_detector_falls_back_and_says_why():
    r = resolve(figure(), dl=(61.0, 168.0))            # depth ~0.05: that is the shoulder, not the armpit
    assert r.left.source is LandmarkSource.SILHOUETTE_APEX
    assert not r.left_evidence.detector.plausible
    assert any("depth" in s for s in r.left_evidence.detector.reasons)


def test_detector_on_the_wrong_side_is_rejected():
    r = resolve(figure(), dl=(162.0, 229.0))
    assert not r.left_evidence.detector.plausible
    assert any("left of the torso centre" in s for s in r.left_evidence.detector.reasons)
    assert r.left.source is LandmarkSource.SILHOUETTE_APEX


def test_detector_off_the_silhouette_is_rejected():
    r = resolve(figure(), dr=(215.0, 213.0))
    assert not r.right_evidence.detector.plausible
    assert any("foreground" in s for s in r.right_evidence.detector.reasons)


def test_garbage_detector_value_does_not_raise():
    r = resolve(figure(), dl="abc", dr=(float("nan"), 3.0))
    assert not r.left_evidence.detector.plausible and not r.right_evidence.detector.plausible
    assert r.method is ResolutionMethod.SILHOUETTE_APEX


# ---------------------------------------------------------------------------------------------- unresolved cases
def test_arms_clamped_and_no_detector_is_unresolved():
    r = resolve(figure(apex_l=None, apex_r=None), dl=None, dr=None)
    assert r.method is ResolutionMethod.UNRESOLVED and not r.resolved and r.points is None
    assert r.left.source is LandmarkSource.UNRESOLVED and r.left.x is None


def test_arms_clamped_but_detector_plausible_still_resolves_by_detector():
    r = resolve(figure(apex_l=None, apex_r=None))
    assert r.method is ResolutionMethod.DETECTOR and r.resolved
    assert any("no arm/torso gap" in s for s in r.left_evidence.apex.reasons)


def test_one_side_unresolved_makes_the_whole_result_unresolved():
    r = resolve(figure(apex_l=None), dl=None)           # left: no detector, no gap | right: detector
    assert r.method is ResolutionMethod.UNRESOLVED
    assert r.right.source is LandmarkSource.DETECTOR and r.left.source is LandmarkSource.UNRESOLVED


# ---------------------------------------------------------------------------------------------- the avatar #2 lesson
def test_dark_crease_is_not_mistaken_for_the_armpit():
    # a 2 px crease of dark skin (sum 24, i.e. > 6 but < 36) from row 170 to 210; the true gap opens at row 210
    img = figure(apex_l=210, apex_r=210, alpha=False, crease_rows=(170, 210))
    r = resolve(img, dl=None, dr=None)
    assert r.diagnostics["foreground_source"].startswith("brightness>6")
    assert r.resolved and abs(r.left.y - 210) <= 2 and abs(r.right.y - 210) <= 2, (r.left.y, r.right.y)
    # for contrast: with the old threshold of 36 the crease reads as an arm/torso gap ~40 rows too high
    arr = np.array(img)
    arr[:, :, :3][arr[:, :, :3].sum(axis=2) <= 36] = 0
    r36 = resolve(Image.fromarray(arr, "RGBA"), dl=None, dr=None)
    assert r36.left_evidence.apex.y is not None and r36.left_evidence.apex.y < 180, r36.left_evidence.apex.y


# ---------------------------------------------------------------------------------------------- inputs / model
def test_alpha_channel_is_used_when_it_carries_a_silhouette():
    r = resolve(figure(alpha=True), dl=None, dr=None)
    assert r.diagnostics["foreground_source"] == "alpha" and r.resolved


def test_pil_and_ndarray_inputs_agree():
    a = resolve(figure(as_pil=True), dl=None, dr=None)
    b = resolve(figure(as_pil=False), dl=None, dr=None)
    assert (a.left.x, a.left.y, a.right.x, a.right.y) == (b.left.x, b.left.y, b.right.x, b.right.y)


def test_shoulders_in_normalised_coordinates_give_unresolved_not_an_exception():
    r = resolve_underarm_landmarks(figure(), (0.2, 0.3), (0.8, 0.3), (61.0, 229.0), (162.0, 213.0))
    assert r.method is ResolutionMethod.UNRESOLVED
    assert any("pixel coordinates" in n for n in r.notes)


def test_empty_black_image_gives_unresolved_not_an_exception():
    r = resolve_underarm_landmarks(Image.new("RGB", (W, H)), LEFT_SH, RIGHT_SH, None, None)
    assert r.method is ResolutionMethod.UNRESOLVED and "foreground is empty" in r.notes


def test_unreadable_image_gives_unresolved_not_an_exception():
    r = resolve_underarm_landmarks(object(), LEFT_SH, RIGHT_SH, None, None)
    assert r.method is ResolutionMethod.UNRESOLVED and r.notes


def test_revision_and_json_round_trip():
    r = resolve(figure(), dl=None)
    assert r.revision == LANDMARK_REVISION == "underarm-v2"
    d = json.loads(json.dumps(r.to_dict()))
    assert d["revision"] == "underarm-v2" and d["method"] == "mixed"
    assert d["left"]["source"] == "silhouette_apex" and d["right"]["source"] == "detector"
    assert set(d["left_evidence"]) == {"detector", "apex", "disagreement_y_px"}


def test_result_is_immutable():
    r = resolve(figure())
    try:
        r.left = None
    except Exception:
        return
    raise AssertionError("result should be frozen")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:                          # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)