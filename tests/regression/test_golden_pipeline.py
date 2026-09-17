"""
Regression suite for the V1 try-on pipeline. Covers the 7 garments
already visually reviewed (Test 2). Tests real behavior at each
stage — not just "did it crash."

Visual regression uses stored reference renders, not arbitrary
pixel-difference thresholds — per the fresh_3/fresh_7 stale-render
incident, a threshold check against a bad reference is worse than
no check at all. Each reference must be freshly generated and
explicitly verified before being captured as a fixture.

Run: python -m pytest tests/regression/test_golden_pipeline.py -v
"""

import os
import numpy as np
import pytest
from PIL import Image

from tryon.avatar_representation import build_avatar_representation
from tryon.garment_representation import build_garment_representation
from tryon.fitting_engine import FittingEngine, UnsupportedGarmentError
from tryon.compositor import Compositor

AVATAR_USER_ID = "15bacab3-5873-401b-9c9a-52f8b3eeeaf7"
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

# The 7 reviewed garments, with their KNOWN, currently-true values —
# not assumed, but what Test 2 actually found. fresh_4's MINOR_ARTIFACT
# status is preserved deliberately, not hidden.
GARMENTS = [
    {"id": "ac7fffb7-2e3a-40f8-894a-0e85fc245373", "label": "golden_polo",
     "shoulders_available": True, "known_status": "PASS",
     "artifact_type": None, "expected_diagonal_hem": True},
    {"id": "5c22b2ea-fedc-4dbb-b649-fc645774d011", "label": "fresh_3",
     "shoulders_available": True, "known_status": "PASS",
     "artifact_type": None, "expected_diagonal_hem": True},
    {"id": "d5d05081-fdb8-4104-8b56-53bb06e2bc93", "label": "fresh_4",
     "shoulders_available": True, "known_status": "MINOR_ARTIFACT",
     "artifact_type": "garment_body_shear", "expected_diagonal_hem": True},
    {"id": "3a46afa7-e179-4d8b-a007-ef9af08fe27b", "label": "fresh_5",
     "shoulders_available": True, "known_status": "PASS",
     "artifact_type": None, "expected_diagonal_hem": True},
    {"id": "d7256fcb-c37d-4a44-88de-d593ad61f20e", "label": "fresh_7",
     "shoulders_available": True, "known_status": "PASS",
     "artifact_type": None, "expected_diagonal_hem": True},
    {"id": "42b7a010-c4fa-4b8f-8ed2-10c3088967b5", "label": "fresh_8",
     "shoulders_available": True, "known_status": "PASS",
     "artifact_type": None, "expected_diagonal_hem": True},
    {"id": "831098a3-8778-480f-9a11-aa9df826d0f2", "label": "fresh_9",
     "shoulders_available": True, "known_status": "PASS",
     "artifact_type": None, "expected_diagonal_hem": True},
]


@pytest.fixture(scope="module")
def avatar():
    """Built once per test run — same avatar for every garment,
    matching how the actual generalization test worked."""
    return build_avatar_representation()


@pytest.fixture(params=GARMENTS, ids=[g["label"] for g in GARMENTS])
def garment_case(request):
    return request.param


class TestRepresentation:
    """Checks 1-3: representation succeeds, underarms exist,
    shoulders_available matches the known value."""

    def test_garment_representation_succeeds(self, garment_case):
        rep = build_garment_representation(garment_case["id"], AVATAR_USER_ID)
        assert rep is not None

    def test_underarms_exist(self, garment_case):
        rep = build_garment_representation(garment_case["id"], AVATAR_USER_ID)
        assert rep.left_underarm is not None
        assert rep.right_underarm is not None
        assert len(rep.left_underarm) == 2
        assert len(rep.right_underarm) == 2

    def test_shoulders_available_matches_known_value(self, garment_case):
        rep = build_garment_representation(garment_case["id"], AVATAR_USER_ID)
        assert rep.shoulders_available == garment_case["shoulders_available"], (
            f"{garment_case['label']}: expected shoulders_available="
            f"{garment_case['shoulders_available']}, got {rep.shoulders_available}. "
            f"This is a real regression if it changed — re-diagnose, don't "
            f"just update the expected value."
        )


class TestAvatarRepresentation:
    def test_avatar_underarms_exist(self, avatar):
        assert avatar.left_underarm is not None
        assert avatar.right_underarm is not None

    def test_avatar_shoulders_exist(self, avatar):
        assert avatar.left_shoulder is not None
        assert avatar.right_shoulder is not None


class TestFittingEngine:
    """Checks 4-5, 9: fitting succeeds, layer dimensions correct,
    no NaN/Inf in geometry."""

    def test_fitting_succeeds_or_raises_correctly(self, avatar, garment_case):
        garment = build_garment_representation(garment_case["id"], AVATAR_USER_ID)

        if garment_case["shoulders_available"]:
            result = FittingEngine.fit(garment, avatar)
            assert result is not None
        else:
            with pytest.raises(UnsupportedGarmentError):
                FittingEngine.fit(garment, avatar)

    def test_layer_dimensions_match_avatar(self, avatar, garment_case):
        if not garment_case["shoulders_available"]:
            pytest.skip("No fitting result to check for unsupported garments")
        garment = build_garment_representation(garment_case["id"], AVATAR_USER_ID)
        result = FittingEngine.fit(garment, avatar)

        aw, ah = avatar.image.size
        assert result.upper_layer.shape == (ah, aw, 4)
        assert result.lower_layer.shape == (ah, aw, 4)

    def test_no_nan_or_inf_in_layers(self, avatar, garment_case):
        if not garment_case["shoulders_available"]:
            pytest.skip("No fitting result to check for unsupported garments")
        garment = build_garment_representation(garment_case["id"], AVATAR_USER_ID)
        result = FittingEngine.fit(garment, avatar)

        for layer, name in [(result.upper_layer, "upper"), (result.lower_layer, "lower")]:
            assert np.isfinite(layer.astype(np.float64)).all(), (
                f"{garment_case['label']}: {name}_layer contains NaN/Inf"
            )


class TestCompositor:
    """Checks 6-8: compositing succeeds, dimensions unchanged,
    output isn't accidentally identical to the original avatar."""

    def test_composite_succeeds(self, avatar, garment_case):
        if not garment_case["shoulders_available"]:
            pytest.skip("No fitting result to composite for unsupported garments")
        garment = build_garment_representation(garment_case["id"], AVATAR_USER_ID)
        result = FittingEngine.fit(garment, avatar)
        final = Compositor.composite(avatar.image, result)
        assert final is not None

    def test_output_dimensions_unchanged(self, avatar, garment_case):
        if not garment_case["shoulders_available"]:
            pytest.skip("No fitting result to composite for unsupported garments")
        garment = build_garment_representation(garment_case["id"], AVATAR_USER_ID)
        result = FittingEngine.fit(garment, avatar)
        final = Compositor.composite(avatar.image, result)
        assert final.size == avatar.image.size

    def test_output_differs_from_original_avatar(self, avatar, garment_case):
        if not garment_case["shoulders_available"]:
            pytest.skip("No fitting result to composite for unsupported garments")
        garment = build_garment_representation(garment_case["id"], AVATAR_USER_ID)
        result = FittingEngine.fit(garment, avatar)
        final = Compositor.composite(avatar.image, result)

        final_arr = np.array(final.convert("RGB"))
        avatar_arr = np.array(avatar.image.convert("RGB"))
        diff = np.abs(final_arr.astype(int) - avatar_arr.astype(int)).sum()
        assert diff > 1000, (
            f"{garment_case['label']}: output is suspiciously close to the "
            f"unmodified avatar — garment may not have been composited at all"
        )


class TestVisualRegression:
    """
    Check 10: golden/reference output comparison. Uses STORED
    reference images, not arbitrary thresholds computed on the fly.

    First run: no fixtures exist yet, so these are skipped with a
    clear message. A separate, explicit capture step (not part of
    normal test runs) creates the fixtures after human visual
    verification — never auto-generated as a side effect of testing.
    """

    def test_matches_reference_render(self, avatar, garment_case):
        if not garment_case["shoulders_available"]:
            pytest.skip("No render to compare for unsupported garments")

        fixture_path = os.path.join(FIXTURE_DIR, f"{garment_case['label']}_reference.png")
        if not os.path.exists(fixture_path):
            pytest.skip(
                f"No reference fixture yet for {garment_case['label']} at "
                f"{fixture_path}. Capture one explicitly (after human visual "
                f"verification) via the separate capture script — do not "
                f"auto-generate fixtures as a side effect of running tests."
            )

        garment = build_garment_representation(garment_case["id"], AVATAR_USER_ID)
        result = FittingEngine.fit(garment, avatar)
        current = Compositor.composite(avatar.image, result)

        reference = Image.open(fixture_path).convert("RGB")
        current_rgb = current.convert("RGB")

        assert current_rgb.size == reference.size, (
            f"{garment_case['label']}: current render size {current_rgb.size} "
            f"differs from reference {reference.size}"
        )

        current_arr = np.array(current_rgb).astype(int)
        reference_arr = np.array(reference).astype(int)
        diff = np.abs(current_arr - reference_arr)
        mean_diff = diff.mean()

        # Generous threshold — this is a smoke test for "did something
        # change substantially," not a pixel-perfect check. Large
        # deviations warrant human re-review, not auto-pass/fail on
        # a tight threshold (same caution as the fresh_3 lesson).
        assert mean_diff < 15.0, (
            f"{garment_case['label']}: current render differs substantially "
            f"from stored reference (mean diff={mean_diff:.2f}). This may be "
            f"a real regression, OR the reference itself may be stale — "
            f"regenerate and visually re-verify before concluding either way."
        )