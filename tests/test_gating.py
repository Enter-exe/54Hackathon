"""Tests for pipeline/gating.py -- the real-truck-photo vs
placeholder/graphic classifier. This exists because a dealer-branded
"Photos Coming Soon" graphic was found scoring as a real photo (and even
as a top "rust" hit) after surviving the hash-based placeholder filter.
"""
import numpy as np
import pytest
import torch

from pipeline.extract_embeddings import MODEL_NAME, load_model
from pipeline.gating import filter_real_photos, load_gate_text_embeddings, real_photo_probability

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def model_and_preprocess():
    return load_model(DEVICE)


@pytest.fixture(scope="session")
def gate_embeddings(model_and_preprocess):
    model, _ = model_and_preprocess
    return load_gate_text_embeddings(model, DEVICE, MODEL_NAME)


def test_load_gate_text_embeddings_shapes_and_norms(gate_embeddings):
    real_emb, not_real_embs = gate_embeddings
    assert real_emb.shape == (512,)
    assert not_real_embs.shape[1] == 512
    assert not_real_embs.shape[0] >= 1
    assert np.linalg.norm(real_emb) == pytest.approx(1.0, abs=1e-4)


def test_real_photo_probability_high_for_real_prompt_itself(gate_embeddings):
    real_emb, not_real_embs = gate_embeddings
    prob = real_photo_probability(real_emb, real_emb, not_real_embs, logit_scale=100.0)
    assert prob > 0.9


def test_real_photo_probability_low_for_placeholder_prompt(gate_embeddings):
    real_emb, not_real_embs = gate_embeddings
    placeholder_emb = not_real_embs[0]
    prob = real_photo_probability(placeholder_emb, real_emb, not_real_embs, logit_scale=100.0)
    assert prob < 0.1


def test_filter_real_photos_mask_matches_expectation(gate_embeddings):
    real_emb, not_real_embs = gate_embeddings
    embeddings = np.stack([real_emb, not_real_embs[0]])
    mask = filter_real_photos(embeddings, real_emb, not_real_embs, logit_scale=100.0)
    assert list(mask) == [True, False]
