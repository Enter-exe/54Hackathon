"""Load and rank train-set comparable sales with cosine similarity."""

from pathlib import Path

import numpy as np

from pipeline.data_io import read_listings_csv


def load_comparables(index_path: Path, listings_path: Path) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Load the comparable index and canonical listing records keyed by ad ID."""
    with np.load(index_path, allow_pickle=True) as archive:
        index = {key: archive[key] for key in archive.files}

    embeddings = index["embeddings"]
    ad_ids = index["ad_id"].astype(str)
    if embeddings.shape[0] != len(ad_ids):
        raise ValueError("index embeddings and ad_id must contain the same number of rows")
    index["ad_id"] = ad_ids

    listings = read_listings_csv(listings_path)
    listings_by_id = {
        str(row["ad_id"]): row for row in listings.to_dict(orient="records")
    }
    return index, listings_by_id


def find_comparables(query_embedding: np.ndarray, index: dict[str, np.ndarray], listings_by_id: dict[str, dict], k: int = 3) -> list[dict]:
    """Return the k most similar train listings in UI-ready form."""
    query = np.asarray(query_embedding, dtype=np.float32)
    norm = np.linalg.norm(query)
    if norm == 0:
        raise ValueError("query embedding must be nonzero")
    query /= norm
    similarities = index["embeddings"] @ query
    order = np.argsort(similarities)[::-1][:k]
    results = []
    for position in order:
        ad_id = str(index["ad_id"][position])
        row = listings_by_id[ad_id]
        results.append(
            {
                "ad_id": ad_id,
                "similarity": float(similarities[position]),
                "price": float(row["price"]),
                "year": int(row["year"]),
                "make_name": row["make_name"],
                "model_name": row["model_name"],
                "source_url": row["source_url"],
                "thumbnail_path": row["image_paths"][0],
            }
        )
    return results
