import numpy as np

from pipeline.appraise import AppraisalEngine, artifacts_ready


CONDITION = {
    "rust": {"probability": 0.1, "flag": False},
    "body_damage": {"probability": 0.2, "flag": False},
    "tire_wear": {"probability": 0.8, "flag": True},
    "interior_wear": {"probability": 0.3, "flag": False},
}


def _engine():
    listings = {
        ad_id: {
            "ad_id": ad_id,
            "price": price,
            "year": year,
            "make_name": "MAKE",
            "model_name": ad_id.upper(),
            "source_url": f"https://example.com/{ad_id}",
            "image_paths": [f"images/{ad_id}.jpg"],
        }
        for ad_id, price, year in zip("abc", [10_000, 20_000, 30_000], [2020, 2021, 2022])
    }
    return AppraisalEngine(
        model=object(),
        preprocess=object(),
        device="cpu",
        price_models=object(),
        condition_text_pairs=object(),
        condition_thresholds=object(),
        photo_gate=object(),
        comparable_index={
            "embeddings": np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]),
            "ad_id": np.array(["a", "b", "c"]),
        },
        listings_by_id=listings,
    )


def test_appraise_returns_complete_accepted_result(monkeypatch):
    # Catches bypassing confidence adjustment, condition output, or real comparable ranking.
    paths = ["truck-a.jpg", "truck-b.jpg", "truck-c.jpg"]
    monkeypatch.setattr(
        "pipeline.appraise.gate_images",
        lambda *args: {
            "accepted": True,
            "confidence": "high",
            "usable_paths": paths,
            "rejected": [],
            "warnings": [],
            "reasons": [],
        },
    )
    monkeypatch.setattr(
        "pipeline.appraise.pooled_listing_embedding",
        lambda *args, **kwargs: np.array([1.0, 0.0]),
    )
    monkeypatch.setattr(
        "pipeline.appraise.predict_quantiles",
        lambda *args: np.array([[24_000.0, 30_000.0, 37_000.0]]),
    )
    monkeypatch.setattr(
        "pipeline.appraise.assess_images", lambda *args, **kwargs: CONDITION
    )

    result = _engine().appraise(paths)

    assert result == {
        "accepted": True,
        "price_low": 24_000.0,
        "price_median": 30_000.0,
        "price_high": 37_000.0,
        "confidence": "high",
        "warnings": [],
        "reasons": [],
        "condition": CONDITION,
        "comparables": [
            {
                "ad_id": "a",
                "similarity": 1.0,
                "price": 10_000.0,
                "year": 2020,
                "make_name": "MAKE",
                "model_name": "A",
                "source_url": "https://example.com/a",
                "thumbnail_path": "images/a.jpg",
            },
            {
                "ad_id": "b",
                "similarity": 0.8,
                "price": 20_000.0,
                "year": 2021,
                "make_name": "MAKE",
                "model_name": "B",
                "source_url": "https://example.com/b",
                "thumbnail_path": "images/b.jpg",
            },
            {
                "ad_id": "c",
                "similarity": 0.0,
                "price": 30_000.0,
                "year": 2022,
                "make_name": "MAKE",
                "model_name": "C",
                "source_url": "https://example.com/c",
                "thumbnail_path": "images/c.jpg",
            },
        ],
    }


def test_appraise_returns_complete_rejected_result(monkeypatch):
    # Catches rejected uploads leaking prices or downstream-only fields.
    monkeypatch.setattr(
        "pipeline.appraise.gate_images",
        lambda *args: {
            "accepted": False,
            "confidence": "low",
            "usable_paths": [],
            "rejected": [{"path": "motorcycle.jpg", "reasons": ["not_truck"]}],
            "warnings": ["some_photos_rejected"],
            "reasons": ["no_usable_truck_photos"],
        },
    )

    result = _engine().appraise(["motorcycle.jpg"])

    assert result == {
        "accepted": False,
        "price_low": None,
        "price_median": None,
        "price_high": None,
        "confidence": "low",
        "warnings": ["some_photos_rejected"],
        "reasons": ["no_usable_truck_photos"],
        "condition": {},
        "comparables": [],
    }


def test_appraise_rejects_when_photo_filter_removes_every_embedding(monkeypatch):
    # Catches passing None into price prediction when the secondary photo gate removes all images.
    monkeypatch.setattr(
        "pipeline.appraise.gate_images",
        lambda *args: {
            "accepted": True,
            "confidence": "high",
            "usable_paths": ["truck.jpg"],
            "rejected": [],
            "warnings": [],
            "reasons": [],
        },
    )
    monkeypatch.setattr(
        "pipeline.appraise.pooled_listing_embedding", lambda *args, **kwargs: None
    )

    result = _engine().appraise(["truck.jpg"])

    assert result == {
        "accepted": False,
        "price_low": None,
        "price_median": None,
        "price_high": None,
        "confidence": "low",
        "warnings": [],
        "reasons": ["no_usable_truck_photos"],
        "condition": {},
        "comparables": [],
    }


def test_artifacts_ready_requires_every_build_output(tmp_path):
    # Catches declaring readiness when any one build artifact is missing.
    required = [
        "data/processed/listings_clean.csv",
        "data/processed/splits.json",
        "data/processed/listing_embeddings.npz",
        "data/processed/comparables_index.npz",
        "data/processed/condition_tags.csv",
        "data/processed/condition_calibration.json",
        "artifacts/price_model/price_models.joblib",
        "artifacts/price_model/price_metrics.json",
    ]
    assert artifacts_ready(tmp_path) is False

    for missing in required:
        root = tmp_path / missing.replace("/", "-")
        for relative in required:
            if relative != missing:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
        assert artifacts_ready(root) is False

    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert artifacts_ready(tmp_path) is True
