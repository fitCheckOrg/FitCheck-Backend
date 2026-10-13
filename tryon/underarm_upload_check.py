"""
tryon/underarm_upload_check.py

The upload-time underarm check: runs the same segmentation + underarm resolution the try-on uses, on the photo that
was just processed, and applies the upload gate to the result.

    decision = check_avatar_photo(enhanced_bytes, pose_keypoints, user_id=user_id)

Modes (environment variable UNDERARM_GATE_MODE, or the `mode` argument):
    off      do nothing, return None
    shadow   (DEFAULT) run the check, log one JSON line, never block the upload
    enforce  as shadow, but raise FitCheckException(422, UNDERARM_NOT_DETECTED) when the decision is RETAKE

The check FAILS OPEN: if it cannot run (no pose keypoints, model error, anything unexpected) it logs and returns
None, in every mode. A broken checker must not block uploads. Only a clean RETAKE decision in enforce mode raises.
"""

import io
import json
import os
import time
from typing import Optional

from PIL import Image

from shared.exceptions import FitCheckException
from shared.logger import get_logger
from tryon.underarm_upload_gate import GateAction, GateDecision, evaluate_underarm_result

logger = get_logger(__name__)

MODE_ENV_VAR = "UNDERARM_GATE_MODE"
MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_ENFORCE = "enforce"
_MODES = (MODE_OFF, MODE_SHADOW, MODE_ENFORCE)


def get_gate_mode(mode: Optional[str] = None) -> str:
    """Explicit argument, else the environment variable, else shadow. Unknown values fall back to shadow."""
    raw = mode if mode is not None else os.environ.get(MODE_ENV_VAR, MODE_SHADOW)
    value = str(raw).strip().lower()
    if value not in _MODES:
        logger.warning("Unknown %s=%r - using %r", MODE_ENV_VAR, raw, MODE_SHADOW)
        return MODE_SHADOW
    return value


def check_avatar_photo(image, pose_keypoints, user_id: Optional[str] = None,
                       mode: Optional[str] = None) -> Optional[GateDecision]:
    """
    image: PNG/JPEG bytes or a PIL image - the PROCESSED avatar photo (the one whose keypoints these are).
    Returns the GateDecision, or None when the check was off or could not run.
    Raises FitCheckException only for a RETAKE decision in enforce mode.
    """
    mode = get_gate_mode(mode)
    if mode == MODE_OFF:
        return None

    started = time.perf_counter()
    try:
        if not pose_keypoints:
            logger.warning("underarm_gate skipped: no pose keypoints (user=%s)", user_id)
            return None
        if isinstance(image, (bytes, bytearray)):
            image = Image.open(io.BytesIO(image))
        image = image.convert("RGB")                       # same conversion the try-on applies to the stored photo

        from tryon.avatar_representation import build_avatar_representation_from_image   # heavy imports: only when used
        rep = build_avatar_representation_from_image(image, pose_keypoints)
        result = rep.resolve_underarms()
        decision = evaluate_underarm_result(result)
    except Exception:
        logger.exception("underarm_gate errored; failing open (user=%s mode=%s)", user_id, mode)
        return None

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    payload = {
        "mode": mode, "user_id": user_id, "elapsed_ms": elapsed_ms,
        "would_block": decision.action is GateAction.RETAKE,
        "decision": decision.to_dict(), "result": result.to_dict(),
    }
    line = "underarm_gate " + json.dumps(payload, default=str)
    if decision.action is GateAction.ACCEPT:
        logger.info(line)
    else:
        logger.warning(line)

    if mode == MODE_ENFORCE and decision.action is GateAction.RETAKE:
        raise FitCheckException(decision.user_message, code=decision.code, status_code=422)
    return decision