"""Tests for ui/appraisal.py.

Skipped when the trained price model isn't present -- like the rest of the
pipeline's outputs, it's a locally-generated artifact (gitignored, not
committed), so a fresh checkout without having run the pipeline scripts
yet shouldn't fail here.
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ui.appraisal import PRICE_MODEL_PATH, find_comparables, load_resources, run_appraisal

pytestmark = pytest.mark.skipif(
    not PRICE_MODEL_PATH.exists(), reason="trained price model not present; run the pipeline scripts first"
)


@pytest.fixture(scope="module")
def resources():
    return load_resources()


def test_load_resources_returns_none_when_price_model_missing(tmp_path):
    result = load_resources(price_model_path=tmp_path / "nope.joblib")
    assert result is None


def test_run_appraisal_rejects_dark_photo_without_touching_price_model(resources, tmp_path):
    dark_path = tmp_path / "dark.jpg"
    Image.new("RGB", (64, 64), color=(0, 0, 0)).save(dark_path)

    result = run_appraisal([str(dark_path)], resources)

    assert result["gate"]["accepted"] is False
    assert result["condition"] is None
    assert result["price"] is None
    assert result["comparables"] is None


def test_run_appraisal_full_pipeline_on_a_real_listing_photo(resources):
    # any already-downloaded real truck photo works; skip if none present locally
    images = sorted(Path("data/images").glob("*/*.webp"))
    if not images:
        pytest.skip("no local scraped images present")
    sample = [str(p) for p in images[:1]]

    result = run_appraisal(sample, resources)

    assert result["gate"]["accepted"] is True
    assert set(result["price"].keys()) == {
        "accepted", "price_low", "price_median", "price_high", "confidence", "warnings", "reasons",
    }
    assert result["price"]["price_low"] <= result["price"]["price_median"] <= result["price"]["price_high"]
    assert not np.isnan(result["price"]["price_median"])

    assert result["comparables"] is not None
    assert len(result["comparables"]) == 3
    for comp in result["comparables"]:
        assert set(comp.keys()) == {"ad_id", "year", "make_name", "model_name", "price", "similarity"}
    # sorted by similarity, most similar first
    sims = [c["similarity"] for c in result["comparables"]]
    assert sims == sorted(sims, reverse=True)


def test_find_comparables_returns_k_nearest_by_cosine_similarity(resources):
    index = resources["comparables_index"]
    # querying with an exact training embedding should return itself as the top-1 match
    query = index["embeddings"][0]

    results = find_comparables(query, index, k=3)

    assert len(results) == 3
    assert results[0]["ad_id"] == str(index["ad_id"][0])
    assert results[0]["similarity"] == pytest.approx(1.0, abs=1e-4)
