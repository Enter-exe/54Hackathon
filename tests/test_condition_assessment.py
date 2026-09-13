"""Tests for pipeline/condition_assessment.py.

Uses the real CLIP model against tiny synthetic images. The core risk in
this script isn't "does CLIP work" (covered by test_extract_embeddings.py)
but the zero-shot scoring logic itself: does the problem/ok softmax sum to
1, does max-aggregation across images actually pick up a single bad photo
instead of diluting it, and does a listing with no usable images get
skipped instead of crashing the run.
"""
import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from pipeline.condition_assessment import (
    ATTRIBUTES,
    assess_images,
    calibrate_thresholds,
    load_text_embeddings,
    run,
    score_embedding,
)
from pipeline.data_io import write_listings_csv
from pipeline.extract_embeddings import load_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def model_and_preprocess():
    return load_model(DEVICE)


@pytest.fixture(scope="session")
def text_pairs(model_and_preprocess):
    model, _ = model_and_preprocess
    return load_text_embeddings(model, DEVICE)


@pytest.fixture
def sample_image(tmp_path):
    def _make(name: str, color=(200, 30, 30)) -> str:
        path = tmp_path / name
        Image.new("RGB", (64, 64), color=color).save(path, format="JPEG")
        return str(path)

    return _make


def test_load_text_embeddings_shapes_and_norms(text_pairs):
    assert set(text_pairs.keys()) == {a["name"] for a in ATTRIBUTES}
    for problem_emb, ok_emb in text_pairs.values():
        assert problem_emb.shape == (512,)
        assert np.linalg.norm(problem_emb) == pytest.approx(1.0, abs=1e-4)
        assert np.linalg.norm(ok_emb) == pytest.approx(1.0, abs=1e-4)


def test_score_embedding_probabilities_sum_to_one(text_pairs):
    rng = np.random.default_rng(0)
    fake_embedding = rng.normal(size=512).astype(np.float32)
    fake_embedding /= np.linalg.norm(fake_embedding)

    scores = score_embedding(fake_embedding, text_pairs, logit_scale=100.0)

    assert set(scores.keys()) == {a["name"] for a in ATTRIBUTES}
    for p in scores.values():
        assert 0.0 <= p <= 1.0


def test_score_embedding_identical_to_problem_prompt_scores_near_one(text_pairs):
    name = ATTRIBUTES[0]["name"]
    problem_emb, _ = text_pairs[name]
    scores = score_embedding(problem_emb, text_pairs, logit_scale=100.0)
    assert scores[name] > 0.9  # an embedding identical to the "problem" prompt should score as problem


def test_assess_images_no_valid_paths_returns_none(model_and_preprocess, text_pairs, tmp_path):
    model, preprocess = model_and_preprocess
    missing = [str(tmp_path / "nope.jpg")]
    assert assess_images(missing, model, preprocess, DEVICE, text_pairs) is None


def test_assess_images_without_thresholds_returns_probability_only(model_and_preprocess, text_pairs, sample_image):
    model, preprocess = model_and_preprocess
    result = assess_images([sample_image("a.jpg")], model, preprocess, DEVICE, text_pairs)
    assert result is not None
    assert set(result.keys()) == {a["name"] for a in ATTRIBUTES}
    for v in result.values():
        assert 0.0 <= v["probability"] <= 1.0
        assert "flag" not in v  # no thresholds given -> no flag decision made


def test_assess_images_with_thresholds_sets_flag(model_and_preprocess, text_pairs, sample_image):
    model, preprocess = model_and_preprocess
    result = assess_images([sample_image("a.jpg")], model, preprocess, DEVICE, text_pairs)
    probs = {name: v["probability"] for name, v in result.items()}
    # threshold exactly at each attribute's own score: it must flag itself (>=), and a threshold
    # far above 1.0 must never flag
    at_own_score = {name: p for name, p in probs.items()}
    above_max = {name: 2.0 for name in probs}

    flagged_at_own = assess_images([sample_image("a.jpg")], model, preprocess, DEVICE, text_pairs, thresholds=at_own_score)
    flagged_above = assess_images([sample_image("a.jpg")], model, preprocess, DEVICE, text_pairs, thresholds=above_max)

    assert all(v["flag"] for v in flagged_at_own.values())
    assert not any(v["flag"] for v in flagged_above.values())


def test_calibrate_thresholds_uses_requested_percentile():
    probs = {"rust": np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])}
    thresholds = calibrate_thresholds(probs, percentile=80.0)
    assert thresholds["rust"] == pytest.approx(np.percentile(probs["rust"], 80.0))


def test_assess_images_max_aggregation_not_diluted_by_clean_photos(model_and_preprocess, text_pairs, sample_image):
    """A single bad photo among several clean ones should still drive the listing's score up,
    not get averaged away -- this is the whole reason max (not mean) aggregation was chosen."""
    model, preprocess = model_and_preprocess
    attr_name = ATTRIBUTES[0]["name"]
    problem_emb, _ = text_pairs[attr_name]

    # real files so the path.exists() check passes; content is irrelevant since embed_images is stubbed below
    paths = [sample_image("p1.jpg"), sample_image("p2.jpg"), sample_image("p3.jpg")]

    import pipeline.condition_assessment as ca

    def fake_embed_images(image_paths, *_args, **_kwargs):
        rows = np.zeros((len(image_paths), 512), dtype=np.float32)
        rows[0] = problem_emb  # first photo looks exactly like the problem prompt
        for i in range(1, len(image_paths)):
            rows[i] = -problem_emb  # the rest look strongly "ok"-like
        return rows

    orig = ca.embed_images
    ca.embed_images = fake_embed_images
    try:
        multi_result = assess_images(paths, model, preprocess, DEVICE, text_pairs)
    finally:
        ca.embed_images = orig

    assert multi_result is not None
    assert multi_result[attr_name]["probability"] > 0.9


def test_run_end_to_end(model_and_preprocess, sample_image, tmp_path, monkeypatch):
    # About run()'s I/O/schema plumbing, not photo-realism judgment (test_gating.py's job) --
    # the synthetic solid-color fixture images correctly fail the real-photo gate, so bypass it.
    monkeypatch.setattr("pipeline.condition_assessment.gate_mask", lambda embs, gate: np.ones(len(embs), dtype=bool))

    data_dir = tmp_path / "data"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True)

    listings = pd.DataFrame(
        [
            {"ad_id": "1", "image_paths": [sample_image("truck1.jpg", (200, 30, 30))]},
            {"ad_id": "2", "image_paths": [sample_image("truck2.jpg", (30, 30, 200))]},
            {"ad_id": "3", "image_paths": [str(tmp_path / "missing.jpg")]},  # must be skipped
        ]
    )
    write_listings_csv(listings, processed_dir / "listings_clean.csv")
    with open(processed_dir / "splits.json", "w") as f:
        json.dump({"train": ["1", "2"], "val": [], "test": []}, f)

    run(data_dir)

    out_df = pd.read_csv(processed_dir / "condition_tags.csv")
    assert set(out_df["ad_id"]) == {1, 2}  # ad_id 3 skipped: no usable images
    for attr in ATTRIBUTES:
        assert f"{attr['name']}_prob" in out_df.columns
        assert f"{attr['name']}_flag" in out_df.columns
        assert out_df[f"{attr['name']}_flag"].dtype == bool

    with open(processed_dir / "condition_calibration.json") as f:
        calibration = json.load(f)
    assert calibration["percentile"] == pytest.approx(80.0)
    assert set(calibration["thresholds"].keys()) == {a["name"] for a in ATTRIBUTES}
