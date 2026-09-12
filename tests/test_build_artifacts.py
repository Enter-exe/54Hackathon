import json

import joblib
import numpy as np
import pandas as pd
from PIL import Image

from pipeline.build_artifacts import build_artifacts


EXPECTED_NAMES = {
    "listings",
    "splits",
    "embeddings",
    "comparables",
    "condition_tags",
    "condition_calibration",
    "price_models",
    "price_metrics",
}


def _write_sales_fixture(dataset_dir, count=20):
    rows = []
    for index in range(count):
        item_id = f"SALE{index:03d}"
        image_path = dataset_dir / "images" / item_id / "truck.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=(index, index, index)).save(image_path)
        rows.append(
            {
                "item_id": item_id,
                "sale_price_usd_including_buyer_premium": 10_000 + index * 100,
                "year": 2020,
                "make": "Make",
                "model": "Model",
                "state": "OK",
                "image_paths": f"images/{item_id}/truck.jpg",
            }
        )
    pd.DataFrame(rows).to_csv(dataset_dir / "truck_sales_100.csv", index=False)


def test_build_artifacts_writes_and_returns_complete_contract(tmp_path, monkeypatch):
    """Removing a stage or its output must fail the one-command build contract."""
    repo_root = tmp_path / "repo"
    dataset_dir = repo_root / "data" / "commercial_truck_sales_100"
    dataset_dir.mkdir(parents=True)
    _write_sales_fixture(dataset_dir)
    calls = []

    def fake_embeddings(data_dir):
        calls.append("embeddings")
        processed_dir = data_dir / "processed"
        np.savez(processed_dir / "listing_embeddings.npz", ad_ids=np.array(["SALE001"]), embeddings=np.zeros((1, 2)))
        np.savez(
            processed_dir / "comparables_index.npz",
            embeddings=np.zeros((1, 2)),
            ad_id=np.array(["SALE001"]),
            price=np.array([10_000.0]),
            year=np.array([2020]),
            make_name=np.array(["Make"]),
            model_name=np.array(["Model"]),
        )

    def fake_condition(data_dir):
        calls.append("condition")
        processed_dir = data_dir / "processed"
        pd.DataFrame({"ad_id": ["SALE001"]}).to_csv(processed_dir / "condition_tags.csv", index=False)
        (processed_dir / "condition_calibration.json").write_text(json.dumps({"thresholds": {}}))

    def fake_training(embeddings, listings, splits, out_dir, n_estimators):
        calls.append("training")
        assert n_estimators == 5
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"quantiles": [0.1, 0.5, 0.9], "models": {}}, out_dir / "price_models.joblib")
        (out_dir / "price_metrics.json").write_text(json.dumps({"val": {}, "test": {}}))

    monkeypatch.setattr("pipeline.extract_embeddings.run", fake_embeddings)
    monkeypatch.setattr("pipeline.condition_assessment.run", fake_condition)
    monkeypatch.setattr("modeling.train_price.run_training", fake_training)

    outputs = build_artifacts(repo_root, n_estimators=5)

    assert set(outputs) == EXPECTED_NAMES
    assert all(path.exists() for path in outputs.values())
    assert calls == ["embeddings", "condition", "training"]
