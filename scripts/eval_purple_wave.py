"""Score the trained price model against 50 real, completed Purple Wave box
truck sales (data/commercial_truck_sales_100) -- an independent honesty
check the model has never seen, from a different marketplace AND a
different price mechanism (actual auction hammer price including buyer's
premium, not a seller's asking price like Commercial Truck Trader/
TruckPaper). Uses the exact same run_appraisal() pipeline the live demo
uses, not a re-implementation.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from ui.appraisal import load_resources, run_appraisal

DATA_DIR = Path("data/commercial_truck_sales_100")


def main():
    df = pd.read_csv(DATA_DIR / "truck_sales_100.csv")
    box_trucks = df[df["truck_type"] == "box_truck"].copy()
    print(f"Loaded {len(box_trucks)} completed box-truck sales")

    resources = load_resources()
    if resources is None:
        raise SystemExit("Price model not found -- run the training pipeline first.")

    rows = []
    for _, row in box_trucks.iterrows():
        image_paths = [
            str(DATA_DIR / p) for p in str(row["image_paths"]).split("|") if p
        ]
        image_paths = [p for p in image_paths if Path(p).exists()]
        if not image_paths:
            print(f"  SKIP {row['item_id']}: no downloaded images found on disk")
            continue

        result = run_appraisal(image_paths, resources)
        true_price = float(row["sale_price_usd_including_buyer_premium"])

        if not result["gate"]["accepted"]:
            reasons = sorted({r for item in result["gate"]["rejected"] for r in item["reasons"]})
            print(f"  REJECTED {row['item_id']} (true=${true_price:,.0f}): {reasons}")
            rows.append({
                "item_id": row["item_id"], "year": row["year"], "make": row["make"],
                "true_price": true_price, "accepted": False,
            })
            continue

        price = result["price"]
        rows.append({
            "item_id": row["item_id"],
            "year": row["year"],
            "make": row["make"],
            "true_price": true_price,
            "accepted": True,
            "price_low": price["price_low"],
            "price_median": price["price_median"],
            "price_high": price["price_high"],
            "confidence": price["confidence"],
            "in_range": price["price_low"] <= true_price <= price["price_high"],
        })

    out_df = pd.DataFrame(rows)
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    out_df.to_csv(out_dir / "purple_wave_eval.csv", index=False)

    accepted = out_df[out_df["accepted"]]
    n_rejected = len(out_df) - len(accepted)
    print(f"\n{len(accepted)}/{len(out_df)} accepted by the input gate ({n_rejected} rejected)")

    if len(accepted) == 0:
        print("Nothing accepted -- can't compute price metrics.")
        return

    y_true = accepted["true_price"].to_numpy()
    y_pred = accepted["price_median"].to_numpy()
    coverage = accepted["in_range"].mean()
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    summary = {
        "n_total": len(out_df),
        "n_accepted": len(accepted),
        "n_rejected": n_rejected,
        "median_r2_vs_real_sold_price": float(r2),
        "median_mae_vs_real_sold_price": float(mae),
        "interval_coverage_vs_real_sold_price": float(coverage),
    }
    (out_dir / "purple_wave_eval_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\n(for reference, on CTT's own held-out test split: median_r2=0.432, "
          f"median_mae=$14,191, interval_coverage=80.7%)")


if __name__ == "__main__":
    main()
