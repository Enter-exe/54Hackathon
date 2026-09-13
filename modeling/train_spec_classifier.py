"""Train a make classifier on the same frozen CLIP embeddings used for pricing.

Motivation: the UI's "predicted specs" originally used a raw nearest-neighbor
lookup against comparables_index.npz, which measured ~50% make-accuracy on
held-out test listings and specifically confuses visually similar classes
like Ford Econoline vs Chevrolet Express (both full-size cutaway vans) --
real, reported misclassification. A trained, class-weighted classifier can
learn an actual decision boundary between such classes instead of just
trusting whichever single training example happens to be nearest in
embedding space.

Logistic regression (not LightGBM) on top of frozen CLIP embeddings is the
standard "linear probe" approach -- CLIP's embedding space is specifically
trained to be close to linearly separable for this kind of downstream task.

C=100 (much weaker regularization than sklearn's default C=1.0) was picked
after actually sweeping it: at C=1.0, class_weight="balanced" gave fair-ish
but very low-confidence predictions (mean top-1 probability ~0.17 across
11-14 classes, barely above uniform) and only 49% accuracy. Train accuracy
saturates near 100% at high C (a classic overfitting signature), but val
and test accuracy hold steady in the high-80s%/mid-70s% rather than
declining -- i.e. the embedding space is separable enough that the extra
capacity isn't fitting noise, it's fitting real signal that C=1.0's default
regularization was suppressing.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


def load_data(embeddings_path, listings_path, splits_path, label_column="make_name"):
    with np.load(embeddings_path, allow_pickle=False) as cache:
        ad_ids = cache["ad_ids"].astype(str)
        embeddings = cache["embeddings"]

    listings = pd.read_parquet(listings_path)
    listing_ids = listings["ad_id"].astype(str)
    labels_by_id = pd.Series(listings[label_column].to_numpy(), index=listing_ids)
    # scraped make_name has inconsistent casing (e.g. "Isuzu" vs "ISUZU") -- normalize
    labels = np.array([str(v).upper() for v in labels_by_id.loc[ad_ids].to_numpy()])

    splits = json.loads(Path(splits_path).read_text())
    split_of = {str(i): name for name, ids in splits.items() for i in ids}
    split_arr = np.array([split_of.get(a, "unknown") for a in ad_ids])

    return embeddings, labels, split_arr


def train_and_evaluate(embeddings, labels, split_arr, out_dir, min_class_count=5, C=100.0):
    train_mask = split_arr == "train"
    test_mask = split_arr == "test"

    # classes with too few training examples can't be learned meaningfully;
    # exclude them from evaluation rather than silently reporting misleading
    # per-class scores for a class the model had ~1-2 examples of.
    class_counts = pd.Series(labels[train_mask]).value_counts()
    rare_classes = set(class_counts[class_counts < min_class_count].index)

    encoder = LabelEncoder()
    y_all = encoder.fit_transform(labels)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=C)
    clf.fit(embeddings[train_mask], y_all[train_mask])

    eval_mask = test_mask & ~np.isin(labels, list(rare_classes))
    y_pred = clf.predict(embeddings[eval_mask])
    y_true = y_all[eval_mask]

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "n_classes_total": len(encoder.classes_),
        "n_classes_evaluated": len(set(labels[eval_mask])),
        "rare_classes_excluded_from_eval": sorted(rare_classes),
        "n_train": int(train_mask.sum()),
        "n_test_evaluated": int(eval_mask.sum()),
        "n_test_excluded_rare": int(test_mask.sum() - eval_mask.sum()),
    }
    present_labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(
        y_true, y_pred, labels=present_labels, target_names=encoder.classes_[present_labels], zero_division=0
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": clf, "label_encoder": encoder}, out_dir / "make_classifier.joblib")
    (out_dir / "make_classifier_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "make_classifier_report.txt").write_text(report)
    return metrics, report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings", default="data/processed/listing_embeddings.npz")
    ap.add_argument("--listings", default="data/processed/listings_clean.parquet")
    ap.add_argument("--splits", default="data/processed/splits.json")
    ap.add_argument("--out-dir", default="artifacts/spec_classifier")
    ap.add_argument("--min-class-count", type=int, default=5)
    ap.add_argument("--C", type=float, default=100.0)
    args = ap.parse_args()

    embeddings, labels, split_arr = load_data(args.embeddings, args.listings, args.splits)
    metrics, report = train_and_evaluate(embeddings, labels, split_arr, args.out_dir, args.min_class_count, args.C)
    print(json.dumps(metrics, indent=2))
    print(report)


if __name__ == "__main__":
    main()
