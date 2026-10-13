"""
tryon/underarm_upload_check_test.py

Run:  python -m tryon.underarm_upload_check_test
Runs the real AvatarRepresentation + resolver + gate. Only the SCHP model call is replaced by masks taken from a
synthetic figure, so these tests check the wiring and the modes, not the segmentation model.
Needs tryon/underarm_landmark_resolver_test.py next to it (for the figure helper).
"""
import contextlib
import io
import json
import logging

import numpy as np
from PIL import Image

import tryon.avatar_representation as rep_mod
from shared.exceptions import FitCheckException
from tryon.underarm_landmark_resolver_test import figure, LEFT_SH, RIGHT_SH, W, H
from tryon.underarm_upload_check import check_avatar_photo, get_gate_mode, MODE_ENV_VAR
from tryon.underarm_upload_gate import GateAction, RETAKE_CODE

LO, HI, Y_SH = 62, 161, 160
KP = {"left_shoulder": {"x": LEFT_SH[0] / W, "y": LEFT_SH[1] / H},
      "right_shoulder": {"x": RIGHT_SH[0] / W, "y": RIGHT_SH[1] / H}}


def _photo(**kw):
    return figure(alpha=False, **kw).convert("RGB")


def _fake_schp(img):
    """Torso label 2 on columns LO..HI (rows Y_SH..329); label 3 where the figure is foreground outside them."""
    a = np.array(img).sum(axis=2) > 6
    pm = np.zeros(a.shape, np.int64)
    pm[Y_SH:330, LO:HI + 1] = 2
    arms = a.copy(); arms[:, LO:HI + 1] = False; arms[:Y_SH] = False
    pm[arms] = 3
    return pm


class _Capture(logging.Handler):
    def __init__(self): super().__init__(); self.records = []
    def emit(self, r): self.records.append(r)


def run(img, mode, kp=KP, **kw):
    cap = _Capture(); lg = logging.getLogger("tryon.underarm_upload_check"); lg.addHandler(cap); lg.setLevel(logging.DEBUG)
    rep_mod._run_schp_segmentation = _fake_schp
    try:
        out = check_avatar_photo(img, kp, user_id="u1", mode=mode, **kw)
        return out, None, cap.records
    except FitCheckException as e:
        return None, e, cap.records
    finally:
        lg.removeHandler(cap)


def _gate_lines(records):
    return [json.loads(r.getMessage().split("underarm_gate ", 1)[1]) for r in records if r.getMessage().startswith("underarm_gate {")]


@contextlib.contextmanager
def detector(left_row=None, right_row=None):
    """Pin the mask-based detector's answer (a row, or None) so these tests do not depend on the real detector."""
    import tryon.avatar_torso_landmarks_test as det
    saved = det.find_semantic_underarm
    det.find_semantic_underarm = lambda tm, am, w, side: (
        (left_row, []) if side == "left" and left_row is not None else
        (right_row, []) if side == "right" and right_row is not None else (None, []))
    try:
        yield
    finally:
        det.find_semantic_underarm = saved


GOLDEN_ROWS = (229, 213)          # the rows the real detector gave on the golden avatar


def test_good_photo_shadow_is_accepted_and_logged_as_one_json_line():
    with detector(*GOLDEN_ROWS):
        out, err, recs = run(_photo(), "shadow")
    assert err is None and out.action is GateAction.ACCEPT and out.method == "detector"
    lines = _gate_lines(recs)
    assert len(lines) == 1 and lines[0]["mode"] == "shadow" and lines[0]["user_id"] == "u1"
    assert lines[0]["would_block"] is False and lines[0]["elapsed_ms"] >= 0
    assert lines[0]["result"]["revision"] == "underarm-v2" and lines[0]["decision"]["action"] == "accept"
    assert recs[0].levelno == logging.INFO


def _unresolvable(mode):
    # no detector point and arms hugging the torso (no visible gap): neither estimate exists -> RETAKE
    with detector():
        return run(_photo(apex_l=None, apex_r=None), mode)


def test_unresolvable_photo_in_shadow_returns_retake_without_raising():
    out, err, recs = _unresolvable("shadow")
    assert err is None and out.action is GateAction.RETAKE and not out.accepted
    lines = _gate_lines(recs)
    assert lines[0]["would_block"] is True and recs[0].levelno == logging.WARNING


def test_unresolvable_photo_in_enforce_raises_422_with_the_user_message():
    out, err, recs = _unresolvable("enforce")
    assert out is None and isinstance(err, FitCheckException)
    assert err.code == RETAKE_CODE == "UNDERARM_NOT_DETECTED" and err.status_code == 422
    assert "arms slightly away" in str(err)
    assert _gate_lines(recs)[0]["would_block"] is True            # logged before raising


def test_enforce_does_not_raise_for_accept_or_accept_with_warnings():
    with detector(*GOLDEN_ROWS):
        out, err, _ = run(_photo(), "enforce")
    assert err is None and out.action is GateAction.ACCEPT
    with detector():                                               # no detector -> apex on both sides -> warnings
        out, err, _ = run(_photo(), "enforce")
    assert err is None and out.action is GateAction.ACCEPT_WITH_WARNINGS


def test_off_mode_does_nothing_and_never_touches_the_model():
    def boom(img): raise AssertionError("segmentation must not run in off mode")
    rep_mod._run_schp_segmentation = boom
    assert check_avatar_photo(_photo(), KP, mode="off") is None


def test_missing_keypoints_skips_cleanly_even_in_enforce():
    for kp in (None, {}):
        out, err, recs = run(_photo(), "enforce", kp=kp)
        assert out is None and err is None and any("skipped" in r.getMessage() for r in recs)


def test_internal_error_fails_open_even_in_enforce():
    def boom(img): raise RuntimeError("onnx exploded")
    rep_mod._run_schp_segmentation = boom
    try:
        out = check_avatar_photo(_photo(), KP, mode="enforce")
    except FitCheckException:
        raise AssertionError("a broken checker must not block uploads")
    assert out is None


def test_png_bytes_input_works():
    buf = io.BytesIO(); _photo().save(buf, "PNG")
    with detector(*GOLDEN_ROWS):
        out, err, _ = run(buf.getvalue(), "shadow")
    assert err is None and out.action is GateAction.ACCEPT


def test_mode_resolution():
    import os
    saved = os.environ.get(MODE_ENV_VAR)
    try:
        os.environ.pop(MODE_ENV_VAR, None)
        assert get_gate_mode() == "shadow"                         # default
        os.environ[MODE_ENV_VAR] = " ENFORCE "
        assert get_gate_mode() == "enforce"
        os.environ[MODE_ENV_VAR] = "banana"
        assert get_gate_mode() == "shadow"                         # unknown -> safe default
        assert get_gate_mode("off") == "off"                       # explicit argument wins
    finally:
        if saved is None: os.environ.pop(MODE_ENV_VAR, None)
        else: os.environ[MODE_ENV_VAR] = saved


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