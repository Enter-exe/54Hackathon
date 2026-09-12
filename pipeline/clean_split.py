"""Clean scraped CTT listings and produce a grouped train/val/test split.

Reads data/raw/listings.parquet (written by scraper/scrape_ctt.py) plus the
downloaded images in data/images/<ad_id>/, and:
  1. drops listings with no usable price
  2. drops price outliers (outside the 1st-99th percentile)
  3. validates each downloaded image (opens, non-trivial size) and drops
     any that are corrupt or that are a shared stock/placeholder image
     (same file reused across many unrelated listings, detected by hashing
     rather than by knowing the placeholder image ahead of time)
  4. drops listings left with zero valid images
  5. dedupes by ad_id
  6. writes a single grouped 80/10/10 train/val/test split by ad_id

No k-fold split here: the encoder used downstream is a frozen pretrained
model (never trained on this data), so there's no Stage1->Stage2 leakage
path that a held-out split would need to guard against.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image

PLACEHOLDER_MIN_LISTINGS = 5  # a hash reused across >= this many listings is treated as a stock/placeholder image
MIN_IMAGE_BYTES = 2048


def valid_image_paths(ad_id, images_dir: Path) -> list[Path]:
    listing_dir = images_dir / str(ad_id)
    if not listing_dir.exists():
        return []
    paths = []
    for p in sorted(listing_dir.glob("*.webp")):
        if p.stat().st_size < MIN_IMAGE_BYTES:
            continue
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception:
            continue
        paths.append(p)
    return paths


def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    images_dir = data_dir / "images"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_dir / "raw" / "listings.parquet")
    n0 = len(df)

    df = df.drop_duplicates(subset="ad_id", keep="first")
    n_dedup = len(df)

    df = df[df["price"].notna() & (df["price"] > 0)]
    n_price = len(df)

    lo, hi = df["price"].quantile([0.01, 0.99])
    df = df[(df["price"] >= lo) & (df["price"] <= hi)].copy()
    n_outlier = len(df)

    print("Validating images and detecting shared placeholder images...")
    listing_images = {ad_id: valid_image_paths(ad_id, images_dir) for ad_id in df["ad_id"]}

    hash_to_ad_ids: dict[str, set] = {}
    path_hash_cache: dict[Path, str] = {}
    for ad_id, paths in listing_images.items():
        for p in paths:
            h = file_hash(p)
            path_hash_cache[p] = h
            hash_to_ad_ids.setdefault(h, set()).add(ad_id)
    placeholder_hashes = {h for h, ids in hash_to_ad_ids.items() if len(ids) >= PLACEHOLDER_MIN_LISTINGS}
    if placeholder_hashes:
        print(f"  found {len(placeholder_hashes)} likely placeholder/stock image(s), shared across listings")

    final_images = {}
    for ad_id, paths in listing_images.items():
        kept = [p for p in paths if path_hash_cache[p] not in placeholder_hashes]
        final_images[ad_id] = kept

    df.loc[:, "image_paths"] = df["ad_id"].map(lambda a: [str(p) for p in final_images.get(a, [])])
    df.loc[:, "n_images"] = df["image_paths"].map(len)
    df = df[df["n_images"] > 0]
    n_final = len(df)

    df.to_parquet(processed_dir / "listings_clean.parquet", index=False)

    ad_ids = df["ad_id"].tolist()
    random.Random(args.seed).shuffle(ad_ids)
    n = len(ad_ids)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    splits = {
        "train": ad_ids[:n_train],
        "val": ad_ids[n_train:n_train + n_val],
        "test": ad_ids[n_train + n_val:],
    }
    with open(processed_dir / "splits.json", "w") as f:
        json.dump(splits, f)

    print(f"\nRows: scraped={n0} -> deduped={n_dedup} -> priced={n_price} "
          f"-> price-outliers-removed={n_outlier} -> has-valid-image={n_final}")
    print(f"Split sizes: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    print(f"Price range kept: ${lo:,.0f} - ${hi:,.0f}")
    print(f"Wrote {processed_dir / 'listings_clean.parquet'} and {processed_dir / 'splits.json'}")


if __name__ == "__main__":
    main()
