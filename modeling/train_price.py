"""Train quantile price models from cached listing-level image embeddings."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, r2_score


SPLIT_NAMES = ("train", "val", "test")
QUANTILES = (0.1, 0.5, 0.9)


def load_training_data(embeddings_path, listings_path, splits_path):
    """Load, validate, and align embeddings, prices, and split membership."""
    with np.load(embeddings_path, allow_pickle=False) as cache:
        missing_arrays = {"ad_ids", "embeddings"} - set(cache.files)
        if missing_arrays:
            raise ValueError(f"embedding cache missing arrays: {sorted(missing_arrays)}")
        ad_ids = cache["ad_ids"].astype(str)
        embeddings = cache["embeddings"]

    if ad_ids.ndim != 1 or embeddings.ndim != 2:
        raise ValueError("ad_ids must be 1D and embeddings must be 2D")
    if len(ad_ids) != len(embeddings):
        raise ValueError("ad_ids and embeddings must contain the same number of rows")
    if len(set(ad_ids)) != len(ad_ids):
        raise ValueError("embedding ad_ids must be unique")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings must contain only finite values")

    listings = pd.read_parquet(listings_path)
    missing_columns = {"ad_id", "price"} - set(listings.columns)
    if missing_columns:
        raise ValueError(f"listings missing columns: {sorted(missing_columns)}")
    listing_ids = listings["ad_id"].astype(str)
    if listing_ids.duplicated().any():
        raise ValueError("listing ad_ids must be unique")
    prices_by_id = pd.Series(
        listings["price"].to_numpy(dtype=float), index=listing_ids
    )
    missing_prices = sorted(set(ad_ids) - set(prices_by_id.index))
    if missing_prices:
        raise ValueError(f"embeddings missing listing prices: {missing_prices[:5]}")
    prices = prices_by_id.loc[ad_ids].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("prices must be positive finite values")

    splits = json.loads(Path(splits_path).read_text())
    if set(splits) != set(SPLIT_NAMES):
        raise ValueError(f"splits must contain exactly {list(SPLIT_NAMES)}")
    split_sets = {name: {str(value) for value in splits[name]} for name in SPLIT_NAMES}
    if any(not ids for ids in split_sets.values()):
        raise ValueError("train, val, and test splits must be non-empty")
    if any(
        split_sets[left] & split_sets[right]
        for i, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[i + 1 :]
    ):
        raise ValueError("split IDs overlap")
    all_split_ids = set().union(*split_sets.values())
    embedding_ids = set(ad_ids)
    if all_split_ids != embedding_ids:
        missing = sorted(embedding_ids - all_split_ids)
        extra = sorted(all_split_ids - embedding_ids)
        raise ValueError(f"split coverage mismatch: missing={missing[:5]}, extra={extra[:5]}")

    return {
        name: (embeddings[np.isin(ad_ids, list(ids))], prices[np.isin(ad_ids, list(ids))])
        for name, ids in split_sets.items()
    }


def train_models(X, y, n_estimators=200):
    """Fit one LightGBM regressor for each requested price quantile."""
    return {
        quantile: LGBMRegressor(
            objective="quantile",
            alpha=quantile,
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            min_child_samples=5,
            reg_lambda=1.0,
            random_state=0,
            n_jobs=1,
            verbosity=-1,
        ).fit(X, y)
        for quantile in QUANTILES
    }


def predict_quantiles(models, X):
    """Return non-crossing q10, q50, and q90 predictions."""
    predictions = np.column_stack([models[q].predict(X) for q in QUANTILES])
    return np.sort(predictions, axis=1)


def evaluate(models, X, y):
    """Calculate range calibration and point/quantile errors."""
    predictions = predict_quantiles(models, X)
    lower, median, upper = predictions.T
    return {
        "pinball_q10": float(mean_pinball_loss(y, lower, alpha=0.1)),
        "pinball_q50": float(mean_pinball_loss(y, median, alpha=0.5)),
        "pinball_q90": float(mean_pinball_loss(y, upper, alpha=0.9)),
        "median_mae": float(mean_absolute_error(y, median)),
        "median_r2": float(r2_score(y, median)),
        "interval_coverage": float(np.mean((y >= lower) & (y <= upper))),
    }


def run_training(
    embeddings_path,
    listings_path,
    splits_path,
    out_dir,
    n_estimators=200,
):
    """Train from on-disk inputs and save the models and held-out metrics."""
    split_data = load_training_data(embeddings_path, listings_path, splits_path)
    models = train_models(*split_data["train"], n_estimators=n_estimators)
    metrics = {
        name: evaluate(models, *split_data[name]) for name in ("val", "test")
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"quantiles": list(QUANTILES), "models": models},
        out_dir / "price_models.joblib",
    )
    (out_dir / "price_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings", default="data/processed/listing_embeddings.npz"
    )
    parser.add_argument(
        "--listings", default="data/processed/listings_clean.parquet"
    )
    parser.add_argument("--splits", default="data/processed/splits.json")
    parser.add_argument("--out-dir", default="artifacts/price_model")
    parser.add_argument("--n-estimators", type=int, default=200)
    args = parser.parse_args()

    metrics = run_training(
        args.embeddings,
        args.listings,
        args.splits,
        args.out_dir,
        n_estimators=args.n_estimators,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
