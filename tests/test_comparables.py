import numpy as np
import pandas as pd
import pytest

from modeling.comparables import find_comparables, load_comparables
from pipeline.data_io import write_listings_csv


def _rows():
    return {
        key: {
            "ad_id": key,
            "price": price,
            "year": 2020,
            "make_name": "MAKE",
            "model_name": key,
            "source_url": f"https://example.com/{key}",
            "image_paths": [f"images/{key}.jpg"],
        }
        for key, price in zip("abc", [10_000, 20_000, 30_000])
    }


def _index():
    return {
        "embeddings": np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]),
        "ad_id": np.array(["a", "b", "c"]),
    }


def test_find_comparables_returns_descending_cosine_matches_with_ui_fields():
    # Catches ascending ranking and ranking against an unnormalized query.
    results = find_comparables(np.array([0.9, 0.1]), _index(), _rows(), k=2)

    assert [item["ad_id"] for item in results] == ["a", "b"]
    assert results[0]["thumbnail_path"] == "images/a.jpg"
    assert set(results[0]) == {
        "ad_id", "similarity", "price", "year", "make_name", "model_name",
        "source_url", "thumbnail_path",
    }


def test_find_comparables_limits_results_to_index_size():
    # Catches indexing beyond the available comparable rows when k is oversized.
    results = find_comparables(np.array([1.0, 0.0]), _index(), _rows(), k=10)

    assert [item["ad_id"] for item in results] == ["a", "b", "c"]


def test_find_comparables_rejects_zero_query_embedding():
    # Catches silent all-zero similarities for an invalid query.
    with pytest.raises(ValueError, match="^query embedding must be nonzero$"):
        find_comparables(np.array([0.0, 0.0]), _index(), _rows())


def test_load_comparables_rejects_index_row_count_mismatch(tmp_path):
    # Catches a corrupt index whose identifiers cannot align to its embeddings.
    listings_path = tmp_path / "listings.csv"
    write_listings_csv(pd.DataFrame(_rows().values()), listings_path)
    index_path = tmp_path / "comparables_index.npz"
    np.savez(index_path, embeddings=np.array([[1.0, 0.0]]), ad_id=np.array(["a", "b"]))

    with pytest.raises(ValueError, match="index embeddings and ad_id must contain the same number of rows"):
        load_comparables(index_path, listings_path)
