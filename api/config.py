from fastapi import APIRouter
from shared.logger import get_logger
from shared.models.responses import SuccessResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/config", tags=["Config"])

# Anchor rules for garment positioning on the canonical avatar pose.
# Tunable here without a Flutter redeploy — this is the whole point
# of serving it as config rather than hardcoding client-side.
#
# width_landmarks: the two landmarks whose distance defines how wide 
#   the garment should be scaled to
# vertical_anchor: the landmark(s) the garment's top edge aligns to
# scale_multiplier: fudge factor — garments usually need to render 
#   slightly wider than the raw anchor-point distance to look right, 
#   not snug to exact bone width. Starting at a reasonable guess, 
#   expect this to be the first thing adjusted once real prototyping 
#   starts.
# ANCHOR_RULES = {
#     "top": {
#         "width_landmarks": ["left_shoulder", "right_shoulder"],
#         "vertical_anchor": ["left_shoulder", "right_shoulder"],
#         "anchor_offset_ratio": 0.08,
#         "scale_multiplier": 1.15
#     },
#     "bottom": {
#         "width_landmarks": ["left_hip", "right_hip"],
#         "vertical_anchor": ["left_hip", "right_hip"],
#         "anchor_offset_ratio": 0.02,
#         "scale_multiplier": 1.10
#     },
#     "shoes": {
#         "width_landmarks": ["left_ankle", "right_ankle"],
#         "vertical_anchor": ["left_ankle", "right_ankle"],
#         "anchor_offset_ratio": 0.05,  # LOWEST CONFIDENCE — 
#                                         # prioritize testing this early
#         "scale_multiplier": 1.20
#     },
#     "dress": {
#         "width_landmarks": ["left_shoulder", "right_shoulder"],
#         "vertical_anchor": ["left_shoulder", "right_shoulder"],
#         "anchor_offset_ratio": 0.04,
#         "scale_multiplier": 1.15
#     }
# }

ANCHOR_RULES = {
    "top": {
        "width_landmarks": ["left_shoulder", "right_shoulder"],
        "vertical_anchor": ["left_shoulder", "right_shoulder"],
        "anchor_offset_ratio": 0.08,
        "scale_multiplier": 1.15
    },
    "top_long_sleeve": {
        "width_landmarks": ["left_shoulder", "right_shoulder"],
        "vertical_anchor": ["left_shoulder", "right_shoulder"],
        "anchor_offset_ratio": 0.08,
        "scale_multiplier": 1.45
    },
    "bottom": {
        "width_landmarks": ["left_hip", "right_hip"],
        "vertical_anchor": ["left_hip", "right_hip"],
        "anchor_offset_ratio": 0.02,
        "scale_multiplier": 1.10
    },
    "shoes": {
        "width_landmarks": ["left_ankle", "right_ankle"],
        "vertical_anchor": ["left_ankle", "right_ankle"],
        "anchor_offset_ratio": 0.05,
        "scale_multiplier": 1.20
    },
    "dress": {
        "width_landmarks": ["left_shoulder", "right_shoulder"],
        "vertical_anchor": ["left_shoulder", "right_shoulder"],
        "anchor_offset_ratio": 0.04,
        "scale_multiplier": 1.15
    }
}
@router.get("/health")
async def health():
    return { "service": "config", "version": "1.0.0", "status": "healthy" }


@router.get("/anchor-rules", response_model=SuccessResponse)
async def get_anchor_rules():
    """
    V1 garment anchoring rules for the canonical avatar pose.
    Static config, not user data — no auth required.
    Covers top/bottom/shoes/dress only. Accessories, jewellery,
    headwear, scarves, belts intentionally excluded from V1 — 
    each needs its own distinct rule, not yet designed.
    """
    return SuccessResponse(data={"anchor_rules": ANCHOR_RULES})