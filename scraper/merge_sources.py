"""Merge the CTT and TruckPaper raw scrapes into one combined listings
parquet, using photo_ids as the unified column name (CTT: bare hex ids
used with a CDN template; TruckPaper: already-resolved full URLs --
scrape_ctt.py's downloader handles both)."""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ctt-jsonl", default="data/raw/scraped_raw.jsonl")
    ap.add_argument("--truckpaper-jsonl", default="data/raw/truckpaper_raw.jsonl")
    ap.add_argument("--out-parquet", default="data/raw/listings.parquet")
    args = ap.parse_args()

    ctt = pd.read_json(args.ctt_jsonl, lines=True).copy()
    ctt["source"] = "ctt"

    tp = pd.read_json(args.truckpaper_jsonl, lines=True)
    tp = tp.rename(columns={"photo_urls": "photo_ids"})

    common_cols = [
        "ad_id", "year", "make_name", "model_name", "price", "mileage",
        "city", "state_code", "class_name", "category_name", "condition",
        "photo_count", "photo_ids", "source",
    ]
    for col in common_cols:
        if col not in ctt.columns:
            ctt[col] = None
        if col not in tp.columns:
            tp[col] = None

    combined = pd.concat([ctt[common_cols], tp[common_cols]], ignore_index=True)
    n_before = len(combined)
    combined = combined.drop_duplicates(subset="ad_id", keep="first")
    n_collisions = n_before - len(combined)

    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.out_parquet, index=False)
    print(f"CTT: {len(ctt)}  TruckPaper: {len(tp)}  ad_id collisions dropped: {n_collisions}")
    print(f"Wrote {len(combined)} combined listings to {args.out_parquet}")
    print(combined["source"].value_counts())


if __name__ == "__main__":
    main()
