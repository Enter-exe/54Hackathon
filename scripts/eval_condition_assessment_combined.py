"""Pool all hand-labeled condition ground-truth rounds
(artifacts/condition_labeling*/labels.json + answer_key.json) into one
combined precision/recall/coverage report, and list every genuine defect
found across all rounds with whether the model actually flagged it.
"""
import json
from pathlib import Path

ATTRS = ["rust", "body_damage", "tire_wear", "interior_wear"]
ROUNDS = ["artifacts/condition_labeling", "artifacts/condition_labeling_round2", "artifacts/condition_labeling_round3"]


def main():
    all_labels, all_answers = {}, {}
    for d in ROUNDS:
        d = Path(d)
        if not (d / "labels.json").exists():
            continue
        labels = json.loads((d / "labels.json").read_text())
        answers = {str(r["ad_id"]): r for r in json.loads((d / "answer_key.json").read_text())}
        all_labels.update(labels)
        all_answers.update(answers)

    print(f"Combined ground-truth set: {len(all_labels)} hand-labeled listings across {len(ROUNDS)} rounds\n")
    print(f"{'attribute':<15} {'n':>4} {'pos':>4} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3} "
          f"{'precision':>10} {'recall':>8} {'accuracy':>9}")
    print("-" * 80)

    defects_found = []
    for attr in ATTRS:
        tp = fp = fn = tn = 0
        for ad_id, truth in all_labels.items():
            if truth.get("exclude"):
                continue
            y_true = truth.get(attr)
            if y_true is None or ad_id not in all_answers:
                continue
            y_pred = bool(all_answers[ad_id][f"{attr}_flag"])
            if y_true == 1:
                defects_found.append((attr, ad_id, y_pred))
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
        precision_s = f"{tp/(tp+fp):.1%}" if (tp + fp) else "n/a"
        recall_s = f"{tp/(tp+fn):.1%}" if (tp + fn) else "n/a"
        accuracy_s = f"{(tp+tn)/n:.1%}" if n else "n/a"
        print(f"{attr:<15} {n:>4} {pos:>4} {tp:>3} {fp:>3} {fn:>3} {tn:>3} "
              f"{precision_s:>10} {recall_s:>8} {accuracy_s:>9}")

    print(f"\nEvery genuine defect found across all rounds, and whether the model caught it:")
    for attr, ad_id, caught in defects_found:
        print(f"  {attr:<15} ad_id={ad_id:<12} model flagged it: {'YES' if caught else 'NO'}")


if __name__ == "__main__":
    main()
