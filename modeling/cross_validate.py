"""5-fold cross-validation over ALL labeled listings (ignores splits.json's
fixed train/val/test partition) to get a stability read on the price model.

Motivation: the official train/val/test split has only ~90 listings in val
and test, small enough that a single before/after comparison (e.g. "does
more scraped data help?") can be dominated by which specific listings
happened to land in which split, rather than a real effect. This pools
everything and averages over 5 different partitions instead.

Each fold further splits its train portion 90/10 into fit/calibration, so
the CQR calibration margin (see modeling/train_price.py) is computed the
same honest way here: never on the same data it's applied to.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from modeling.train_price import (
    CONDITION_COLUMNS,
    apply_margin,
    compute_calibration_margin,
    evaluate,
    predict_quantiles,
    train_models,
)


def load_all_labeled(embeddings_path, listings_path, condition_tags_path=None):
    with np.load(embeddings_path, allow_pickle=False) as cache:
        ad_ids = cache["ad_ids"].astype(str)
        embeddings = cache["embeddings"]

    listings = pd.read_parquet(listings_path)
    listing_ids = listings["ad_id"].astype(str)
    prices_by_id = pd.Series(listings["price"].to_numpy(dtype=float), index=listing_ids)
    prices = prices_by_id.loc[ad_ids].to_numpy(dtype=float)

    if condition_tags_path is not None and Path(condition_tags_path).exists():
        condition_df = pd.read_parquet(condition_tags_path)
        condition_ids = condition_df["ad_id"].astype(str)
        condition_by_id = condition_df.set_index(condition_ids)[list(CONDITION_COLUMNS)]
        condition_features = condition_by_id.loc[ad_ids].to_numpy(dtype=float)
        embeddings = np.concatenate([embeddings, condition_features], axis=1)

    return embeddings, prices


def run_cv(embeddings_path, listings_path, condition_tags_path, n_folds=5, n_estimators=200, target_coverage=0.8, seed=0):
    X, y = load_all_labeled(embeddings_path, listings_path, condition_tags_path)
    n = len(y)
    print(f"Cross-validating on {n} listings, {n_folds} folds")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics = []
    for fold_i, (fit_idx, test_idx) in enumerate(kf.split(X), 1):
        rng = np.random.default_rng(seed + fold_i)
        shuffled = rng.permutation(fit_idx)
        n_cal = max(1, int(0.1 * len(shuffled)))
        cal_idx, train_idx = shuffled[:n_cal], shuffled[n_cal:]

        models = train_models(X[train_idx], y[train_idx], n_estimators=n_estimators)
        margin = compute_calibration_margin(models, X[cal_idx], y[cal_idx], target_coverage=target_coverage)
        metrics = evaluate(models, X[test_idx], y[test_idx], margin=margin)
        metrics["margin"] = margin
        fold_metrics.append(metrics)
        print(f"  fold {fold_i}: R2={metrics['median_r2']:.3f} MAE={metrics['median_mae']:.0f} coverage={metrics['interval_coverage']:.3f}")

    keys = [k for k in fold_metrics[0] if k != "margin"]
    summary = {
        k: {"mean": float(np.mean([m[k] for m in fold_metrics])), "std": float(np.std([m[k] for m in fold_metrics]))}
        for k in keys
    }
    return summary, fold_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings", default="data/processed/listing_embeddings.npz")
    ap.add_argument("--listings", default="data/processed/listings_clean.parquet")
    ap.add_argument("--condition-tags", default="data/processed/condition_tags.parquet")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--target-coverage", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    summary, _ = run_cv(
        args.embeddings, args.listings, args.condition_tags,
        n_folds=args.folds, n_estimators=args.n_estimators,
        target_coverage=args.target_coverage, seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
