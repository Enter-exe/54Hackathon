"""Compare pipeline/condition_assessment.py's predicted flags against
hand-labeled ground truth (artifacts/condition_labeling/labels.json,
produced against artifacts/condition_labeling/manifest.json's contact
sheets, WITHOUT looking at the model's scores while labeling).

This is the first accuracy check this project has ever run against real
condition judgment -- everything in tests/test_condition_assessment.py
checks the scoring mechanics, not whether a flag means anything.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ATTRS = ["rust", "body_damage", "tire_wear", "interior_wear"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label-dir", default="artifacts/condition_labeling")
    args = ap.parse_args()
    label_dir = Path(args.label_dir)

    labels = json.loads((label_dir / "labels.json").read_text())
    answer_key = {str(r["ad_id"]): r for r in json.loads((label_dir / "answer_key.json").read_text())}

    excluded = [ad_id for ad_id, v in labels.items() if v.get("exclude")]
    if excluded:
        print(f"Excluded {len(excluded)} listing(s) with no usable ground-truth photo: {excluded}\n")

    print(f"{'attribute':<15} {'n':>4} {'pos':>4} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3} "
          f"{'precision':>10} {'recall':>8} {'accuracy':>9}")
    print("-" * 80)

    report = {}
    for attr in ATTRS:
        tp = fp = fn = tn = 0
        for ad_id, truth in labels.items():
            if truth.get("exclude"):
                continue
            y_true = truth.get(attr)
            if y_true is None:  # e.g. interior_wear with no cab-interior photo -- neither side has info
                continue
            if ad_id not in answer_key:
                continue
            y_pred = bool(answer_key[ad_id][f"{attr}_flag"])
            if y_true == 1 and y_pred:
                tp += 1
            elif y_true == 0 and y_pred:
                fp += 1
            elif y_true == 1 and not y_pred:
                fn += 1
            else:
                tn += 1

        n = tp + fp + fn + tn
        pos = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        accuracy = (tp + tn) / n if n else float("nan")
        report[attr] = {
            "n": n, "n_positive_in_ground_truth": pos,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "accuracy": accuracy,
        }
        print(f"{attr:<15} {n:>4} {pos:>4} {tp:>3} {fp:>3} {fn:>3} {tn:>3} "
              f"{precision:>10.1%} {recall if recall==recall else float('nan'):>8} {accuracy:>9.1%}"
              if recall == recall else
              f"{attr:<15} {n:>4} {pos:>4} {tp:>3} {fp:>3} {fn:>3} {tn:>3} "
              f"{precision:>10.1%} {'n/a':>8} {accuracy:>9.1%}")

    out_path = label_dir / "eval_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=lambda x: None if x != x else x))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
