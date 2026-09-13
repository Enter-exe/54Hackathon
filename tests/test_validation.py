import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from pipeline.data_io import write_listings_csv
from pipeline.gate_images import BLUR_THRESHOLD, DARK_THRESHOLD, assess_image_quality
from scripts.validate_pipeline import make_blurry, make_dark, validate_pipeline


def test_adversarial_transforms_cross_quality_thresholds(tmp_path):
    # Catches weakening either deterministic transform until its gate no longer fires.
    source = tmp_path / "source.jpg"
    dark_path = tmp_path / "dark.jpg"
    blurry_path = tmp_path / "blurry.jpg"
    checkerboard = (np.indices((256, 256)).sum(axis=0) % 2 * 255).astype(np.uint8)
    Image.fromarray(checkerboard, mode="L").convert("RGB").save(source)

    make_dark(source, dark_path)
    make_blurry(source, blurry_path)

    assert assess_image_quality(dark_path)["brightness"] < DARK_THRESHOLD
    assert assess_image_quality(blurry_path)["sharpness"] < BLUR_THRESHOLD


def test_validate_pipeline_aggregates_six_cases_and_writes_json(tmp_path):
    # Catches omitting a validation case or losing consumer-facing appraisal fields.
    repo_root = tmp_path / "repo"
    truck_path = repo_root / "data/source/truck.jpg"
    fixture_dir = repo_root / "tests/fixtures"
    truck_path.parent.mkdir(parents=True)
    fixture_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64), "white").save(truck_path)
    Image.new("RGB", (64, 64), "red").save(fixture_dir / "motorcycle.jpg")
    Image.new("RGB", (64, 64), "blue").save(fixture_dir / "european_truck.jpg")
    write_listings_csv(
        pd.DataFrame(
            [
                {
                    "ad_id": "TEST-1",
                    "price": 25_000.0,
                    "year": 2020,
                    "make_name": "Make",
                    "model_name": "Model",
                    "image_paths": ["data/source/truck.jpg"],
                    "source_url": "https://example.com/TEST-1",
                }
            ]
        ),
        repo_root / "data/processed/listings_clean.csv",
    )
    (repo_root / "data/processed/splits.json").write_text(
        json.dumps({"train": [], "val": [], "test": ["TEST-1"]})
    )

    accepted = {
        "accepted": True,
        "price_low": 20_000.0,
        "price_median": 25_000.0,
        "price_high": 30_000.0,
        "confidence": "high",
        "warnings": [],
        "reasons": [],
        "condition": {
            "rust": {"probability": 0.8, "flag": True},
            "body_damage": {"probability": 0.1, "flag": False},
        },
        "comparables": [
            {
                "ad_id": "COMP-1",
                "similarity": 0.9,
                "price": 24_000.0,
                "year": 2019,
                "make_name": "Make",
                "model_name": "Other",
                "source_url": "https://example.com/COMP-1",
                "thumbnail_path": "data/source/truck.jpg",
            }
        ],
    }
    rejected = {
        "accepted": False,
        "price_low": None,
        "price_median": None,
        "price_high": None,
        "confidence": "low",
        "warnings": ["some_photos_rejected"],
        "reasons": ["no_usable_truck_photos"],
        "rejected": [{"path": "fixture.jpg", "reasons": ["not_truck"]}],
        "condition": {},
        "comparables": [],
    }

    class FakeEngine:
        def __init__(self):
            self.results = iter(
                [accepted, rejected, rejected, rejected, {**accepted, "confidence": "low"}, accepted]
            )

        def appraise(self, _image_paths):
            return next(self.results)

    report = validate_pipeline(repo_root, engine=FakeEngine())

    assert set(report) == {
        "held_out_truck",
        "dark",
        "blurry",
        "motorcycle",
        "single_photo",
        "non_us_truck",
    }
    assert report["held_out_truck"] == {
        "accepted": True,
        "confidence": "high",
        "reasons": [],
        "price_low": 20_000.0,
        "price_median": 25_000.0,
        "price_high": 30_000.0,
        "condition_flags": {"rust": True, "body_damage": False},
        "comparable_ids": ["COMP-1"],
    }
    assert report["motorcycle"]["accepted"] is False
    assert report["single_photo"]["confidence"] == "low"
    assert json.loads(
        (repo_root / "artifacts/validation_report.json").read_text()
    ) == report
    json.dumps(report, allow_nan=False)
