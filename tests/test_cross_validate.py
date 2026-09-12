import numpy as np
import pandas as pd
import pytest

from modeling.cross_validate import load_all_labeled, run_cv


@pytest.fixture
def synthetic_dataset(tmp_path):
    rng = np.random.default_rng(0)
    n = 60
    ad_ids = np.array([f"truck-{i}" for i in range(n)])
    embeddings = rng.normal(size=(n, 4)).astype(np.float32)
    prices = 20_000 + 4_000 * embeddings[:, 0]

    embeddings_path = tmp_path / "listing_embeddings.npz"
    np.savez(embeddings_path, ad_ids=ad_ids, embeddings=embeddings)

    listings_path = tmp_path / "listings_clean.parquet"
    pd.DataFrame({"ad_id": ad_ids, "price": prices}).to_parquet(listings_path, index=False)

    condition_path = tmp_path / "condition_tags.parquet"
    pd.DataFrame(
        {
            "ad_id": ad_ids,
            "rust_prob": rng.uniform(size=n),
            "body_damage_prob": rng.uniform(size=n),
            "tire_wear_prob": rng.uniform(size=n),
            "interior_wear_prob": rng.uniform(size=n),
        }
    ).to_parquet(condition_path, index=False)

    return embeddings_path, listings_path, condition_path


def test_load_all_labeled_concatenates_condition_features(synthetic_dataset):
    embeddings_path, listings_path, condition_path = synthetic_dataset
    X, y = load_all_labeled(embeddings_path, listings_path, condition_path)
    assert X.shape == (60, 4 + 4)
    assert y.shape == (60,)


def test_load_all_labeled_without_condition_tags(synthetic_dataset):
    embeddings_path, listings_path, _ = synthetic_dataset
    X, y = load_all_labeled(embeddings_path, listings_path, condition_tags_path=None)
    assert X.shape == (60, 4)


def test_run_cv_returns_mean_and_std_for_each_metric(synthetic_dataset):
    embeddings_path, listings_path, condition_path = synthetic_dataset

    summary, fold_metrics = run_cv(
        embeddings_path, listings_path, condition_path, n_folds=3, n_estimators=10
    )

    assert len(fold_metrics) == 3
    for key in ["pinball_q10", "pinball_q50", "pinball_q90", "median_mae", "median_r2", "interval_coverage"]:
        assert key in summary
        assert "mean" in summary[key] and "std" in summary[key]
        assert np.isfinite(summary[key]["mean"])
        assert np.isfinite(summary[key]["std"])
