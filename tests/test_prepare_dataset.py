import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from pipeline.data_io import read_listings_csv
from pipeline.prepare_dataset import prepare_sales_data


def _write_fixture_sales(dataset_dir: Path, count: int = 20) -> None:
    rows = []
    for number in range(1, count + 1):
        ad_id = f"SALE{number:03d}"
        image_path = dataset_dir / "images" / ad_id / "truck.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 20), color=(number, 0, 0)).save(image_path)
        rows.append(
            {
                "item_id": ad_id,
                "year": 2000 + number,
                "make": "Make",
                "model": f"Model {number}",
                "state": "OK",
                "sale_price_usd_including_buyer_premium": float(number),
                "image_paths": f"images/{ad_id}/truck.jpg",
                "source_url": f"https://example.com/{ad_id}",
            }
        )
    pd.DataFrame(rows).to_csv(dataset_dir / "truck_sales_100.csv", index=False)


def test_prepare_sales_data_normalizes_filters_and_splits_fixture_sales(tmp_path):
    repo_root = tmp_path / "repo"
    dataset_dir = repo_root / "data" / "commercial_truck_sales_100"
    processed_dir = repo_root / "data" / "processed"
    second_processed_dir = repo_root / "data" / "processed_again"
    dataset_dir.mkdir(parents=True)
    _write_fixture_sales(dataset_dir)

    clean, splits = prepare_sales_data(dataset_dir, processed_dir, repo_root, seed=0)
    _, repeated_splits = prepare_sales_data(
        dataset_dir, second_processed_dir, repo_root, seed=0
    )

    assert {"ad_id", "price", "year", "make_name", "model_name", "image_paths", "source_url"} <= set(clean)
    assert clean["ad_id"].map(type).eq(str).all()
    assert clean["ad_id"].tolist() == [f"SALE{number:03d}" for number in range(2, 20)]
    assert clean["price"].tolist() == [float(number) for number in range(2, 20)]
    assert clean["make_name"].eq("Make").all()
    assert clean["model_name"].tolist() == [f"Model {number}" for number in range(2, 20)]
    assert clean["state_code"].eq("OK").all()
    assert clean["source_url"].tolist() == [f"https://example.com/SALE{number:03d}" for number in range(2, 20)]
    assert all((repo_root / path).is_file() for paths in clean["image_paths"] for path in paths)
    assert not (set(splits["train"]) & set(splits["val"]))
    assert not (set(splits["train"]) & set(splits["test"]))
    assert not (set(splits["val"]) & set(splits["test"]))
    assert set().union(*map(set, splits.values())) == set(clean["ad_id"])
    assert {name: len(ids) for name, ids in splits.items()} == {"train": 14, "val": 2, "test": 2}
    assert repeated_splits == splits
    assert read_listings_csv(processed_dir / "listings_clean.csv")["image_paths"].map(bool).all()
    assert json.loads((processed_dir / "splits.json").read_text()) == splits


def test_prepare_sales_data_rejects_duplicate_nonpositive_nonfinite_and_imageless_rows(tmp_path):
    repo_root = tmp_path / "repo"
    dataset_dir = repo_root / "data" / "commercial_truck_sales_100"
    processed_dir = repo_root / "data" / "processed"
    dataset_dir.mkdir(parents=True)
    _write_fixture_sales(dataset_dir, count=10)
    sales_path = dataset_dir / "truck_sales_100.csv"
    sales = pd.read_csv(sales_path, dtype={"item_id": str})
    sales.loc[1, "item_id"] = "SALE001"
    sales.loc[2, "sale_price_usd_including_buyer_premium"] = 0
    sales.loc[3, "sale_price_usd_including_buyer_premium"] = np.inf
    corrupt_image_path = dataset_dir / "images" / "SALE005" / "corrupt.jpg"
    corrupt_image_path.write_text("not a JPEG")
    sales.loc[4, "image_paths"] = "images/SALE005/corrupt.jpg"
    sales.to_csv(sales_path, index=False)

    clean, _ = prepare_sales_data(dataset_dir, processed_dir, repo_root)

    assert set(clean["ad_id"]) == {"SALE006", "SALE007", "SALE008", "SALE009"}
