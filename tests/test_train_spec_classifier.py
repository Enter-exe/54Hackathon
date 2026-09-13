import json

import numpy as np
import pandas as pd
import pytest

from modeling.train_spec_classifier import load_data, train_and_evaluate


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Two visually-separable clusters (makes A and B), enough examples per
    class to be learnable, plus one near-singleton class to exercise the
    rare-class exclusion."""
    rng = np.random.default_rng(0)
    n_per_class = 20
    make_a = rng.normal(loc=0.0, scale=0.5, size=(n_per_class, 8))
    make_b = rng.normal(loc=5.0, scale=0.5, size=(n_per_class, 8))
    make_rare = rng.normal(loc=-5.0, scale=0.5, size=(2, 8))
    embeddings = np.concatenate([make_a, make_b, make_rare]).astype(np.float32)
    labels = ["MAKE_A"] * n_per_class + ["make_b"] * n_per_class + ["RARE"] * 2  # mixed case on purpose
    ad_ids = np.array([f"truck-{i}" for i in range(len(labels))])

    embeddings_path = tmp_path / "listing_embeddings.npz"
    np.savez(embeddings_path, ad_ids=ad_ids, embeddings=embeddings)

    listings_path = tmp_path / "listings_clean.parquet"
    pd.DataFrame({"ad_id": ad_ids, "make_name": labels}).to_parquet(listings_path, index=False)

    rng.shuffle(idx := np.arange(len(ad_ids)))
    n_train = int(0.7 * len(idx))
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(
        json.dumps(
            {
                "train": list(ad_ids[idx[:n_train]]),
                "val": [],
                "test": list(ad_ids[idx[n_train:]]),
            }
        )
    )

    return embeddings_path, listings_path, splits_path


def test_load_data_normalizes_make_name_casing(synthetic_dataset):
    embeddings_path, listings_path, splits_path = synthetic_dataset
    embeddings, labels, split_arr = load_data(embeddings_path, listings_path, splits_path)

    assert "MAKE_A" in labels
    assert "MAKE_B" in labels  # "make_b" upper-cased to match "MAKE_A"'s casing
    assert set(split_arr) <= {"train", "val", "test"}


def test_load_data_drops_rows_with_missing_label_for_that_column(tmp_path):
    """e.g. TruckPaper listings have no GVWR class_name -- those rows must be
    excluded from that classifier's training, not turned into a spurious
    'NONE' class."""
    ad_ids = np.array([f"truck-{i}" for i in range(5)])
    embeddings = np.arange(15, dtype=np.float32).reshape(5, 3)
    embeddings_path = tmp_path / "listing_embeddings.npz"
    np.savez(embeddings_path, ad_ids=ad_ids, embeddings=embeddings)

    listings_path = tmp_path / "listings_clean.parquet"
    pd.DataFrame(
        {"ad_id": ad_ids, "class_name": ["CLASS 6", "CLASS 6", None, "CLASS 4", None]}
    ).to_parquet(listings_path, index=False)

    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps({"train": list(ad_ids), "val": [], "test": []}))

    embeddings_out, labels, split_arr = load_data(embeddings_path, listings_path, splits_path, "class_name")

    assert len(labels) == 3  # the two None rows dropped
    assert embeddings_out.shape == (3, 3)
    assert len(split_arr) == 3


def test_train_and_evaluate_separates_well_separated_clusters(synthetic_dataset, tmp_path):
    embeddings_path, listings_path, splits_path = synthetic_dataset
    embeddings, labels, split_arr = load_data(embeddings_path, listings_path, splits_path)

    metrics, report = train_and_evaluate(embeddings, labels, split_arr, tmp_path / "out", min_class_count=5)

    assert metrics["accuracy"] > 0.9  # trivially separable clusters -- should be nearly perfect
    assert "RARE" in metrics["rare_classes_excluded_from_eval"]
    assert (tmp_path / "out" / "make_classifier.joblib").exists()


def test_train_and_evaluate_excludes_rare_classes_from_metrics(synthetic_dataset, tmp_path):
    embeddings_path, listings_path, splits_path = synthetic_dataset
    embeddings, labels, split_arr = load_data(embeddings_path, listings_path, splits_path)

    metrics, _ = train_and_evaluate(embeddings, labels, split_arr, tmp_path / "out", min_class_count=5)

    # every rare-excluded test row must be accounted for -- none silently dropped or double-counted
    assert metrics["n_test_evaluated"] + metrics["n_test_excluded_rare"] == int((split_arr == "test").sum())
