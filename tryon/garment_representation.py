"""
GarmentRepresentation — the first formal production module,
consolidating what has been validated across the research phase:

    find_garment_underarms()          -> underarms (REQUIRED, always attempted)
    get_garment_shoulder_points()     -> shoulders (OPTIONAL, confidence-scored)
    get_binary_mask()                 -> mask / silhouette

Deliberately mirrors AvatarRepresentation's discipline: this class
holds real, validated evidence — it does not invent or guess a
shoulder point when detection fails. shoulders_available=False is
a first-class, expected outcome, not an error state to hide.

No fitting logic, no deformation, no rendering — this class answers
only "what do we know about this garment," the same separation of
concerns the checkpoint calls for.
"""

import io
import requests
import numpy as np
import cv2
from PIL import Image

from tryon.correspondence_shoulders_test import find_garment_underarms
from tryon.garment_shoulder_region_extraction import get_binary_mask
from tryon.piecewise_continuous_test import get_garment_shoulder_points

ALPHA_THRESHOLD = 10


class GarmentRepresentation:
    """
    image: PIL.Image, the garment's clean (background-removed) photo
    mask: np.ndarray[bool], the garment silhouette

    underarms: dict{"left": (x,y), "right": (x,y)} — REQUIRED.
        Validated across 7/7 in-contract garments tonight. If this
        ever fails, treat it as a real pipeline stop, not something
        to paper over (per fresh_5's genuine failure earlier this
        session — a long-sleeve garment outside the V1 contract).

    shoulders: dict{"left": (x,y), "right": (x,y)} or None —
        OPTIONAL. Real on tested textured garments; explicitly
        uncharacterized on plain garments (see checkpoint). Always
        check shoulders_available before using — never assume
        presence.

    shoulders_available: bool — answers only "did the representation
        successfully obtain the shoulder landmarks?" This is
        availability, not confidence — it says nothing about whether
        those landmarks are semantically correct. Always check before
        using .left_shoulder/.right_shoulder.
    """

    def __init__(self, image, mask, underarms, shoulders, shoulders_available):
        self.image = image
        self.mask = mask
        self.underarms = underarms
        self.shoulders = shoulders
        self.shoulders_available = shoulders_available

    @property
    def width(self):
        return self.image.size[0]

    @property
    def height(self):
        return self.image.size[1]

    @property
    def left_underarm(self):
        return self.underarms["left"]

    @property
    def right_underarm(self):
        return self.underarms["right"]

    @property
    def left_shoulder(self):
        if not self.shoulders_available:
            raise ValueError(
                "Shoulder landmarks not available for this garment "
                "(shoulders_available=False). Check shoulders_available "
                "before accessing — do not assume presence."
            )
        return self.shoulders["left"]

    @property
    def right_shoulder(self):
        if not self.shoulders_available:
            raise ValueError(
                "Shoulder landmarks not available for this garment "
                "(shoulders_available=False). Check shoulders_available "
                "before accessing — do not assume presence."
            )
        return self.shoulders["right"]

    def silhouette_contour(self):
        """Real garment outer contour, computed on demand (not cached
        at construction — only needed by some downstream consumers)."""
        contours, _ = cv2.findContours(
            self.mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return max(contours, key=cv2.contourArea)


def build_garment_representation(garment_id, avatar_user_id):
    """
    Fetches the garment and builds a real GarmentRepresentation.

    Underarm detection failure raises — it's a real pipeline stop,
    not something to catch and hide (mirrors AvatarRepresentation's
    "raise rather than guess" discipline for missing pose_keypoints).

    Shoulder detection failure does NOT raise — it sets
    shoulders_available=False and shoulders=None, which is the
    expected, handled outcome for plain garments per tonight's
    characterization.
    """
    from core.storage.supabase_client import supabase
    from shared.exceptions import ClothingItemNotFoundError

    result = supabase.table("closet_items")\
        .select("clean_image_url")\
        .eq("item_id", str(garment_id))\
        .eq("user_id", str(avatar_user_id))\
        .maybe_single()\
        .execute()

    if not result or not result.data:
        raise ClothingItemNotFoundError()

    garment_resp = result.data
    
    garment_bytes = requests.get(garment_resp["clean_image_url"]).content
    mask, image = get_binary_mask(garment_bytes)
    w, h = image.size

    left_ua, right_ua = find_garment_underarms(mask, w)
    if left_ua is None or right_ua is None:
        raise ValueError(
            f"Underarm detection failed for garment {garment_id}. "
            f"This is a real pipeline stop — check whether this garment "
            f"is within the V1 contract (short-sleeve, flat-laid)."
        )
    underarms = {"left": left_ua, "right": right_ua}

    left_sh, right_sh = get_garment_shoulder_points(mask, image, left_ua, right_ua, w)
    if left_sh is not None and right_sh is not None:
        shoulders = {"left": left_sh, "right": right_sh}
        shoulders_available = True
    else:
        shoulders = None
        shoulders_available = False

    return GarmentRepresentation(image, mask, underarms, shoulders, shoulders_available)


if __name__ == "__main__":
    # Sanity check against the golden garment — same discipline as
    # avatar_representation.py's own standalone check
    GOLDEN_GARMENT_ID = "ac7fffb7-2e3a-40f8-894a-0e85fc245373"
    AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"

    rep = build_garment_representation(GOLDEN_GARMENT_ID, AVATAR_USER_ID)
    print(f"Image size: {rep.width} x {rep.height}")
    print(f"Underarms: L={rep.left_underarm}  R={rep.right_underarm}")
    print(f"Shoulders available: {rep.shoulders_available}")
    if rep.shoulders_available:
        print(f"Shoulders: L={rep.left_shoulder}  R={rep.right_shoulder}")
    print("\n>>> GarmentRepresentation built successfully — real evidence held, "
          "nothing guessed, shoulder availability explicit.")