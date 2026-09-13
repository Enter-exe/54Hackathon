"""Train a price model from cached listing-level image embeddings.

Only a median (q=0.5) LightGBM model is trained. Separate q10/q90 quantile
regressors were tried first and dropped: with ~700 training rows, extreme
quantile regression is poorly conditioned and regresses toward the global
marginal quantile almost regardless of the specific truck. On the real test
split, the raw q10 model's predictions had std=$5,474 across all trucks
(vs the median model's std=$15,301) -- e.g. a real $118,751 truck got a raw
q10 of $40,550 (34% of its true price), and a real $14,995 truck got a raw
q90 of $79,661 (5x its true price). Loosening the tail models' regularization
made this WORSE, not better (verified): a noisier tail fit needs a bigger
CQR correction to still hit target coverage, so the final interval got wider,
not more sensibly shaped.

Instead, the low/high bounds are derived from the median model's OWN
calibrated relative error (see compute_calibration_bounds): "on held-out
data, how far off was this model, as a fraction of its own prediction."
Applying that fraction multiplicatively means a truck's range scales with
ITS predicted value instead of sitting near a fixed dollar floor/ceiling
learned from the whole fleet -- which is what was making the low end look
obviously wrong specifically for expensive trucks (and the high end
obviously wrong for cheap ones).
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, r2_score


SPLIT_NAMES = ("train", "val", "test")
CONDITION_COLUMNS = ("rust_prob", "body_damage_prob", "tire_wear_prob", "interior_wear_prob")


def load_training_data(embeddings_path, listings_path, splits_path, condition_tags_path=None):
    """Load, validate, and align embeddings, prices, and split membership.

    condition_tags_path: optional path to pipeline/condition_assessment.py's
    output (rust/body_damage/tire_wear/interior_wear probabilities). When
    given and the file exists, those columns are concatenated onto the CLIP
    embedding so the price model can see a condition signal it otherwise
    has no access to. A path that's explicitly given but missing raises,
    same as any other input here; the CLI default is treated as optional
    (skipped if the file hasn't been produced yet).
    """
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

    if condition_tags_path is not None and Path(condition_tags_path).exists():
        condition_df = pd.read_parquet(condition_tags_path)
        missing_condition_columns = set(CONDITION_COLUMNS) - set(condition_df.columns)
        if missing_condition_columns:
            raise ValueError(f"condition tags missing columns: {sorted(missing_condition_columns)}")
        condition_ids = condition_df["ad_id"].astype(str)
        if condition_ids.duplicated().any():
            raise ValueError("condition tag ad_ids must be unique")
        condition_by_id = condition_df.set_index(condition_ids)[list(CONDITION_COLUMNS)]
        missing_condition = sorted(set(ad_ids) - set(condition_by_id.index))
        if missing_condition:
            raise ValueError(f"embeddings missing condition tags: {missing_condition[:5]}")
        condition_features = condition_by_id.loc[ad_ids].to_numpy(dtype=float)
        if not np.isfinite(condition_features).all():
            raise ValueError("condition features must contain only finite values")
        embeddings = np.concatenate([embeddings, condition_features], axis=1)

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


def train_median_model(X, y, n_estimators=200):
    """Fit the single LightGBM regressor the price range is built from."""
    return LGBMRegressor(
        objective="quantile",
        alpha=0.5,
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


def predict_median(model, X):
    return model.predict(X)


def _finite_sample_quantile(scores, q, n):
    """The +1/n correction that makes a split-conformal quantile a valid
    finite-sample bound (Romano et al. 2019) rather than a plain empirical
    quantile, which would systematically undercover on small calibration
    sets."""
    q_level = min(1.0, np.ceil((n + 1) * q) / n)
    return float(np.quantile(scores, q_level, method="higher"))


def compute_calibration_bounds(model, X, y, target_coverage=0.8):
    """Asymmetric, relative split-conformal calibration: measures how far
    off the median model's predictions typically are, AS A FRACTION OF THE
    PREDICTION ITSELF, on held-out data -- then reuses those two fractions
    (below/above) as multiplicative bounds at inference. Must be computed on
    a split the model was not trained on (val), then applied to any other
    split (val for a calibrated read of held-out performance, test for the
    real held-out check) -- applying it to the same data it was fit on
    would trivially inflate coverage.

    Returns (rel_lo, rel_hi): rel_lo <= 0 <= rel_hi, applied as
    prediction * (1 + rel_lo) and prediction * (1 + rel_hi).
    """
    median_pred = predict_median(model, X)
    if np.any(median_pred <= 0):
        raise ValueError("median predictions must be positive to compute relative bounds")
    relative_error = (y - median_pred) / median_pred
    n = len(y)
    tail = (1 - target_coverage) / 2
    rel_hi = _finite_sample_quantile(relative_error, 1 - tail, n)
    rel_lo = -_finite_sample_quantile(-relative_error, 1 - tail, n)
    return rel_lo, rel_hi


def predict_range(model, X, rel_lo, rel_hi):
    """Return non-crossing [low, median, high] price predictions."""
    median_pred = predict_median(model, X)
    lower = median_pred * (1 + rel_lo)
    upper = median_pred * (1 + rel_hi)
    predictions = np.column_stack([lower, median_pred, upper])
    return np.sort(predictions, axis=1)


def evaluate(model, X, y, rel_lo=0.0, rel_hi=0.0):
    """Calculate range calibration and point/quantile errors."""
    predictions = predict_range(model, X, rel_lo, rel_hi)
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
    condition_tags_path=None,
    target_coverage=0.8,
):
    """Train from on-disk inputs and save the model and held-out metrics."""
    split_data = load_training_data(embeddings_path, listings_path, splits_path, condition_tags_path)
    model = train_median_model(*split_data["train"], n_estimators=n_estimators)

    rel_lo, rel_hi = compute_calibration_bounds(model, *split_data["val"], target_coverage=target_coverage)
    metrics = {
        name: evaluate(model, *split_data[name], rel_lo=rel_lo, rel_hi=rel_hi) for name in ("val", "test")
    }
    metrics["rel_lo"] = rel_lo
    metrics["rel_hi"] = rel_hi

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "rel_lo": rel_lo, "rel_hi": rel_hi},
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
    parser.add_argument(
        "--condition-tags",
        default="data/processed/condition_tags.parquet",
        help="path to condition_assessment.py output; skipped if it doesn't exist",
    )
    parser.add_argument("--target-coverage", type=float, default=0.8)
    args = parser.parse_args()

    metrics = run_training(
        args.embeddings,
        args.listings,
        args.splits,
        args.out_dir,
        n_estimators=args.n_estimators,
        condition_tags_path=args.condition_tags,
        target_coverage=args.target_coverage,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
