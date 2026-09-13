"""Tests for scripts/validate_pipeline.py.

Skipped when the trained price model isn't present, same reasoning as
tests/test_ui_appraisal.py -- these artifacts are locally generated and
gitignored, not committed.
"""
import numpy as np
import pytest
from PIL import Image

from scripts.validate_pipeline import make_blurry, make_dark, summarize, validate_pipeline
from ui.appraisal import PRICE_MODEL_PATH
from pipeline.gate_images import assess_image_quality, DARK_THRESHOLD, BLUR_THRESHOLD

pytestmark = pytest.mark.skipif(
    not PRICE_MODEL_PATH.exists(), reason="trained price model not present; run the pipeline scripts first"
)


def test_make_dark_drops_brightness_below_gate_threshold(tmp_path):
    source = tmp_path / "source.jpg"
    Image.new("RGB", (64, 64), color=(200, 200, 200)).save(source)
    dark_path = tmp_path / "dark.jpg"

    make_dark(source, dark_path)

    assert assess_image_quality(dark_path, DARK_THRESHOLD, BLUR_THRESHOLD)["brightness"] < DARK_THRESHOLD


def test_make_blurry_drops_sharpness_below_gate_threshold(tmp_path):
    checkerboard = (np.indices((64, 64)).sum(axis=0) % 2) * 255
    source = tmp_path / "source.jpg"
    Image.fromarray(checkerboard.astype(np.uint8)).save(source)
    blurry_path = tmp_path / "blurry.jpg"

    make_blurry(source, blurry_path)

    assert assess_image_quality(blurry_path, DARK_THRESHOLD, BLUR_THRESHOLD)["sharpness"] < BLUR_THRESHOLD


def test_summarize_extracts_expected_fields_from_accepted_result():
    fake_result = {
        "gate": {"accepted": True, "confidence": "high", "reasons": []},
        "price": {"price_low": 1.0, "price_median": 2.0, "price_high": 3.0},
        "predicted_make": [{"make_name": "FORD", "probability": 0.9}],
        "condition": {"rust": {"flag": True}, "tire_wear": {"flag": False}},
    }

    summary = summarize(fake_result)

    assert summary == {
        "accepted": True,
        "confidence": "high",
        "reasons": [],
        "price_low": 1.0,
        "price_median": 2.0,
        "price_high": 3.0,
        "predicted_make": "FORD",
        "n_condition_flags": 1,
    }


def test_summarize_handles_rejected_result():
    fake_result = {
        "gate": {"accepted": False, "confidence": "low", "reasons": ["no_usable_truck_photos"]},
        "price": None,
        "predicted_make": None,
        "condition": None,
    }

    summary = summarize(fake_result)

    assert summary["accepted"] is False
    assert summary["price_median"] is None
    assert summary["predicted_make"] is None
    assert summary["n_condition_flags"] is None


def test_validate_pipeline_runs_all_six_cases_and_rejects_bad_inputs(tmp_path):
    report = validate_pipeline(tmp_path)

    assert set(report) == {
        "held_out_truck", "dark", "blurry", "motorcycle", "single_photo", "european_truck",
    }
    assert report["held_out_truck"]["accepted"] is True
    assert report["dark"]["accepted"] is False
    assert report["blurry"]["accepted"] is False
    assert report["motorcycle"]["accepted"] is False
    assert report["single_photo"]["accepted"] is True
    assert report["single_photo"]["confidence"] == "low"
