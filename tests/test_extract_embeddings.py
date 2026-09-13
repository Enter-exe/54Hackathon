"""Tests for pipeline/extract_embeddings.py.

Uses the real CLIP model (loaded once, session-scoped) against tiny
synthetic images -- no real truck photos needed. This catches the class of
bugs that actually broke this script during development: a mismatched
model config that silently degrades embeddings (QuickGELU), pandas
chained-assignment corrupting the split column, and images that fail to
load crashing the whole run instead of being skipped.
"""
import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from pipeline.data_io import write_listings_csv
from pipeline.extract_embeddings import embed_images, load_model, pooled_listing_embedding, run

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def model_and_preprocess():
    return load_model(DEVICE)


@pytest.fixture
def sample_image(tmp_path):
    def _make(name: str, color=(200, 30, 30)) -> str:
        path = tmp_path / name
        Image.new("RGB", (64, 64), color=color).save(path, format="JPEG")
        return str(path)

    return _make


def test_pooled_listing_embedding_empty_list_returns_none(model_and_preprocess):
    model, preprocess = model_and_preprocess
    assert pooled_listing_embedding([], model, preprocess, DEVICE) is None


def test_pooled_listing_embedding_missing_files_returns_none(model_and_preprocess, tmp_path):
    model, preprocess = model_and_preprocess
    missing = [str(tmp_path / "does_not_exist_1.jpg"), str(tmp_path / "does_not_exist_2.jpg")]
    assert pooled_listing_embedding(missing, model, preprocess, DEVICE) is None


def test_pooled_listing_embedding_valid_images_are_unit_norm(model_and_preprocess, sample_image):
    model, preprocess = model_and_preprocess
    paths = [sample_image("a.jpg", (200, 30, 30)), sample_image("b.jpg", (30, 30, 200))]
    pooled = pooled_listing_embedding(paths, model, preprocess, DEVICE)
    assert pooled is not None
    assert pooled.shape == (512,)
    assert not np.isnan(pooled).any()
    assert np.linalg.norm(pooled) == pytest.approx(1.0, abs=1e-4)


def test_pooled_listing_embedding_ignores_missing_paths_mixed_with_valid(model_and_preprocess, sample_image, tmp_path):
    model, preprocess = model_and_preprocess
    paths = [sample_image("a.jpg"), str(tmp_path / "missing.jpg")]
    pooled = pooled_listing_embedding(paths, model, preprocess, DEVICE)
    assert pooled is not None
    assert np.linalg.norm(pooled) == pytest.approx(1.0, abs=1e-4)


def test_embed_images_skips_corrupt_file_without_crashing(model_and_preprocess, sample_image, tmp_path):
    model, preprocess = model_and_preprocess
    corrupt_path = tmp_path / "corrupt.jpg"
    corrupt_path.write_text("this is not an image")
    paths = [sample_image("good.jpg"), str(corrupt_path)]

    result = embed_images(paths, model, preprocess, DEVICE)

    assert result.shape == (2, 512)
    assert np.linalg.norm(result[0]) == pytest.approx(1.0, abs=1e-4)
    assert np.allclose(result[1], 0.0)  # corrupt image left as the zero row, not dropped or crashed on


def test_run_end_to_end(model_and_preprocess, sample_image, tmp_path, monkeypatch):
    # This test is about run()'s I/O/schema plumbing, not photo-realism judgment
    # (that's test_gating.py's job) -- the synthetic solid-color fixture images
    # correctly fail the real-photo gate, so bypass it here.
    monkeypatch.setattr("pipeline.extract_embeddings.gate_mask", lambda embs, gate: np.ones(len(embs), dtype=bool))

    data_dir = tmp_path / "data"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True)

    good_img = sample_image("truck1.jpg", (120, 80, 40))
    listings = pd.DataFrame(
        [
            {
                "ad_id": "1",
                "price": 25000,
                "year": 2020,
                "make_name": "ISUZU",
                "model_name": "NPR",
                "image_paths": [good_img],
            },
            {
                "ad_id": "2",
                "price": 31000,
                "year": 2021,
                "make_name": "FORD",
                "model_name": "E450",
                "image_paths": [good_img],
            },
            {
                "ad_id": "3",  # no valid images -- must be skipped, not crash the run
                "price": 40000,
                "year": 2019,
                "make_name": "HINO",
                "model_name": "195",
                "image_paths": [str(tmp_path / "missing.jpg")],
            },
        ]
    )
    write_listings_csv(listings, processed_dir / "listings_clean.csv")
    with open(processed_dir / "splits.json", "w") as f:
        json.dump({"train": ["1"], "val": ["2"], "test": ["3"]}, f)  # ad_id 3 will get dropped and must be pruned back out

    run(data_dir)

    index = np.load(processed_dir / "comparables_index.npz", allow_pickle=True)
    assert list(index["ad_id"]) == ["1"]  # only the train-split listing goes into the comparables index

    # all-split cache consumed by modeling/train_price.py
    listing_cache = np.load(processed_dir / "listing_embeddings.npz", allow_pickle=False)
    assert set(listing_cache["ad_ids"]) == {"1", "2"}
    assert listing_cache["embeddings"].shape == (2, 512)

    # ad_id 3 was dropped (no valid images) -- it must be pruned from splits.json too, or
    # a consumer cross-checking split coverage against the embeddings (train_price.py) breaks
    with open(processed_dir / "splits.json") as f:
        pruned_splits = json.load(f)
    assert pruned_splits == {"train": ["1"], "val": ["2"], "test": []}
