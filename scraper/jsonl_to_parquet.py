"""Convert the browser-collected JSONL listing dump into data/raw/listings.parquet."""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", default="data/raw/scraped_raw.jsonl")
    ap.add_argument("--out-parquet", default="data/raw/listings.parquet")
    args = ap.parse_args()

    df = pd.read_json(args.in_jsonl, lines=True)
    n0 = len(df)
    df = df.drop_duplicates(subset="ad_id", keep="first")
    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_parquet, index=False)
    print(f"{n0} rows read, {len(df)} unique listings written to {args.out_parquet}")


if __name__ == "__main__":
    main()
