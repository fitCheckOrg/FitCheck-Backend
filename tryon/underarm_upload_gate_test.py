"""
tryon/underarm_upload_gate_test.py

Run:  python -m tryon.underarm_upload_gate_test
Uses the real resolver on the same synthetic figures as underarm_landmark_resolver_test (needs that file next to it).
"""
import json

from tryon.underarm_landmark_resolver_test import figure, resolve
from tryon.underarm_upload_gate import (
    GateAction, RETAKE_CODE, RETAKE_MESSAGE, evaluate_underarm_result,
)


def test_detector_both_sides_is_accepted_even_with_large_disagreement():
    d = evaluate_underarm_result(resolve(figure()))              # golden-like: disagreement +6 / -10 px, as on the real avatar
    assert d.action is GateAction.ACCEPT and d.accepted and d.warnings == () and d.code is None
    assert d.method == "detector" and abs(d.disagreement_y_px[0] - 6.0) <= 1 and abs(d.disagreement_y_px[1] + 10.0) <= 1


def test_one_side_fallback_is_accepted_with_a_warning():
    d = evaluate_underarm_result(resolve(figure(), dl=None))
    assert d.action is GateAction.ACCEPT_WITH_WARNINGS and d.accepted and d.method == "mixed"
    assert len(d.warnings) == 1 and d.warnings[0].startswith("left: used silhouette apex")
    assert "no detector point" in d.warnings[0]


def test_both_sides_apex_gives_two_warnings():
    d = evaluate_underarm_result(resolve(figure(), dl=None, dr=None))
    assert d.action is GateAction.ACCEPT_WITH_WARNINGS and d.method == "silhouette_apex" and len(d.warnings) == 2


def test_implausible_detector_replaced_by_apex_says_why():
    d = evaluate_underarm_result(resolve(figure(), dl=(61.0, 168.0)))
    assert d.action is GateAction.ACCEPT_WITH_WARNINGS and "implausible" in d.warnings[0] and "depth" in d.warnings[0]


def test_unresolved_is_a_retake_with_the_user_safe_message():
    d = evaluate_underarm_result(resolve(figure(apex_l=None, apex_r=None), dl=None, dr=None))
    assert d.action is GateAction.RETAKE and not d.accepted
    assert d.code == RETAKE_CODE == "UNDERARM_NOT_DETECTED" and d.user_message == RETAKE_MESSAGE
    assert any(r.startswith("left apex:") and "no arm/torso gap" in r for r in d.reasons)
    assert any(r.startswith("right detector:") for r in d.reasons)


def test_only_the_unresolved_side_is_explained():
    d = evaluate_underarm_result(resolve(figure(apex_l=None), dl=None))     # left: nothing | right: detector
    assert d.action is GateAction.RETAKE
    assert d.reasons and all(r.startswith("left") for r in d.reasons)


def test_bad_shoulders_retake_carries_the_resolver_note():
    from tryon.underarm_landmark_resolver import resolve_underarm_landmarks
    r = resolve_underarm_landmarks(figure(), (0.2, 0.3), (0.8, 0.3), (61.0, 229.0), (162.0, 213.0))
    d = evaluate_underarm_result(r)
    assert d.action is GateAction.RETAKE and any("pixel coordinates" in x for x in d.reasons)


def test_retake_message_has_no_internal_terms():
    for bad in ("detector", "apex", "silhouette", "mask", "pixel"):
        assert bad not in RETAKE_MESSAGE.lower()


def test_decision_json_round_trip_and_revision():
    for r in (resolve(figure()), resolve(figure(), dl=None), resolve(figure(apex_l=None, apex_r=None), dl=None, dr=None)):
        d = evaluate_underarm_result(r)
        j = json.loads(json.dumps(d.to_dict()))
        assert j["revision"] == "underarm-v2" and j["action"] == d.action.value


def test_decision_is_immutable():
    d = evaluate_underarm_result(resolve(figure()))
    try:
        d.action = GateAction.RETAKE
    except Exception:
        return
    raise AssertionError("decision should be frozen")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS  {name}")
        except Exception as exc:                                  # noqa: BLE001
            failed += 1; print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)