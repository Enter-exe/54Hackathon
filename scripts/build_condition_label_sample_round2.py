"""Round 2 of hand-labeled condition ground truth.

Round 1 (build_condition_label_sample.py) stratified by the OLD flag/clean
split within val+test and found zero real defects in 40 trucks -- useful
for checking false-positive rate, but it never surfaced a single true
positive, so recall (does the model actually catch a truck that IS
damaged?) is still completely unverified.

This round deliberately hunts for a real defect instead of checking
calibration: it pulls the most extreme raw scores per attribute across the
WHOLE dataset (train included -- we're not re-deriving thresholds from
these labels, just checking whether the ranking has any signal at all at
its most confident end) plus the cheapest listings, on the theory that a
low price is the closest available proxy for higher mileage/wear in this
photo-only dataset. Listings already labeled in round 1 are excluded.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from build_condition_label_sample import build_contact_sheet

ATTRS = ["rust", "body_damage", "tire_wear", "interior_wear"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="artifacts/condition_labeling_round2")
    ap.add_argument("--round1-labels", default="artifacts/condition_labeling/labels.json")
    ap.add_argument("--top-k-per-attr", type=int, default=10)
    ap.add_argument("--n-cheapest", type=int, default=10)
    args = ap.parse_args()

    processed_dir = Path(args.data_dir) / "processed"
    out_dir = Path(args.out_dir)

    listings = pd.read_parquet(processed_dir / "listings_clean.parquet")
    tags = pd.read_parquet(processed_dir / "condition_tags.parquet")

    already_labeled = set(json.loads(Path(args.round1_labels).read_text()).keys())
    tags = tags[~tags["ad_id"].astype(str).isin(already_labeled)]

    sample_ids: list = []
    reasons: dict = {}
    for attr in ATTRS:
        top = tags.nlargest(args.top_k_per_attr, f"{attr}_prob")["ad_id"].tolist()
        for ad_id in top:
            reasons.setdefault(ad_id, []).append(f"top-{attr}-score")
        sample_ids += top

    listings_pool = listings[listings["ad_id"].isin(tags["ad_id"])]
    cheapest = listings_pool.nsmallest(args.n_cheapest, "price")["ad_id"].tolist()
    for ad_id in cheapest:
        reasons.setdefault(ad_id, []).append("cheapest-price")
    sample_ids += cheapest

    sample_ids = list(dict.fromkeys(sample_ids))  # dedupe, keep order

    listings_by_id = listings.set_index("ad_id")
    manifest = []
    for ad_id in sample_ids:
        if ad_id not in listings_by_id.index:
            continue
        image_paths = list(listings_by_id.loc[ad_id, "image_paths"])
        sheet_path = out_dir / "contact_sheets" / f"{ad_id}.jpg"
        n_used = build_contact_sheet(image_paths, sheet_path)
        if n_used == 0:
            continue
        manifest.append({
            "ad_id": int(ad_id), "contact_sheet": str(sheet_path), "n_images": n_used,
            "sampled_because": reasons[ad_id],
            "price": float(listings_by_id.loc[ad_id, "price"]),
        })

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    answer_key = tags[tags["ad_id"].isin([m["ad_id"] for m in manifest])].to_dict("records")
    (out_dir / "answer_key.json").write_text(json.dumps(answer_key, indent=2))

    print(f"Sampled {len(manifest)} new listings (top-{args.top_k_per_attr} per attribute + "
          f"{args.n_cheapest} cheapest), excluding {len(already_labeled)} already labeled in round 1.")
    print(f"Wrote {out_dir / 'manifest.json'} (label these, WITHOUT peeking at answer_key.json)")


if __name__ == "__main__":
    main()
