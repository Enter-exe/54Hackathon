"""Normalize the checked-in completed-sale dataset for downstream modeling."""

import argparse
import json
import math
import random
from pathlib import Path

import pandas as pd
from PIL import Image

from pipeline.data_io import write_listings_csv


COLUMN_MAP = {
    "item_id": "ad_id",
    "sale_price_usd_including_buyer_premium": "price",
    "make": "make_name",
    "model": "model_name",
    "state": "state_code",
}


def image_is_valid(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, SyntaxError):
        return False
    return True


def _relative_image_paths(value, dataset_dir: Path, repo_root: Path) -> list[str]:
    if not isinstance(value, str):
        return []
    paths = [dataset_dir / item for item in value.split("|") if item]
    return [
        path.relative_to(repo_root).as_posix()
        for path in paths
        if image_is_valid(path)
    ]


def _split_ids(ids: list[str], seed: int) -> dict[str, list[str]]:
    random.Random(seed).shuffle(ids)
    count = len(ids)
    train_count = round(0.8 * count)
    val_count = round(0.1 * count)
    return {
        "train": ids[:train_count],
        "val": ids[train_count : train_count + val_count],
        "test": ids[train_count + val_count :],
    }


def prepare_sales_data(
    dataset_dir: Path,
    processed_dir: Path,
    repo_root: Path,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Clean checked-in sales, write the CSV/splits contract, and return both."""
    dataset_dir = Path(dataset_dir)
    processed_dir = Path(processed_dir)
    repo_root = Path(repo_root)
    source = pd.read_csv(dataset_dir / "truck_sales_100.csv", dtype={"item_id": str})
    clean = source.rename(columns=COLUMN_MAP)[
        ["ad_id", "price", "year", "make_name", "model_name", "state_code", "image_paths"]
    ].copy()
    clean = clean[clean["ad_id"].notna()].copy()
    clean["ad_id"] = clean["ad_id"].astype(str)
    clean = clean.drop_duplicates(subset="ad_id", keep="first")

    clean["price"] = pd.to_numeric(clean["price"], errors="coerce")
    clean = clean[clean["price"].map(lambda price: math.isfinite(price) and price > 0)].copy()
    lower, upper = clean["price"].quantile([0.01, 0.99])
    clean = clean[clean["price"].between(lower, upper, inclusive="both")].copy()

    clean["image_paths"] = clean["image_paths"].map(
        lambda value: _relative_image_paths(value, dataset_dir, repo_root)
    )
    clean = clean[clean["image_paths"].map(bool)].reset_index(drop=True)

    splits = _split_ids(clean["ad_id"].tolist(), seed)
    write_listings_csv(clean, processed_dir / "listings_clean.csv")
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "splits.json").write_text(json.dumps(splits, indent=2) + "\n")
    return clean, splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="data/commercial_truck_sales_100")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    clean, splits = prepare_sales_data(
        Path(args.dataset_dir), Path(args.processed_dir), Path(args.repo_root), args.seed
    )
    print(f"Wrote {args.processed_dir}/listings_clean.csv ({len(clean)} listings)")
    print("Split sizes: " + " ".join(f"{name}={len(ids)}" for name, ids in splits.items()))


if __name__ == "__main__":
    main()
