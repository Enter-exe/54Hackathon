"""Zero-shot CLIP gate: is this actually a usable photo of a truck?

Two uses:
  - data quality: filter out placeholder/graphic images that survive the
    hash-based placeholder check in clean_split.py. That check only catches
    byte-identical images repeated across listings; a dealer-branded
    "Photos Coming Soon" graphic (address/logo baked in per-dealer) is
    unique enough to slip through, and it was found sitting in the
    embeddings feeding both the price model and condition assessment.
  - inference-time gating (step 6): refuse to price an upload that isn't
    actually a truck photo.
"""
from dataclasses import dataclass

import numpy as np
import open_clip
import torch

REAL_PROMPT = "a real outdoor photograph of a truck"
NOT_REAL_PROMPTS = [
    "a graphic illustration, icon, or placeholder image",
    "a 'photo coming soon' or 'no photo available' placeholder graphic",
    "a vehicle covered by a cloth or blanket, shown as an illustration",
]

DEFAULT_REAL_PHOTO_THRESHOLD = 0.5


@dataclass
class PhotoGate:
    real_emb: np.ndarray
    not_real_embs: np.ndarray
    logit_scale: float
    threshold: float = DEFAULT_REAL_PHOTO_THRESHOLD


def build_gate(model, device: str, model_name: str, threshold: float = DEFAULT_REAL_PHOTO_THRESHOLD) -> PhotoGate:
    real_emb, not_real_embs = load_gate_text_embeddings(model, device, model_name)
    logit_scale = model.logit_scale.exp().item()
    return PhotoGate(real_emb, not_real_embs, logit_scale, threshold)


def load_gate_text_embeddings(model, device: str, model_name: str):
    """Returns (real_embedding, not_real_embeddings) both L2-normalized."""
    tokenizer = open_clip.get_tokenizer(model_name)
    prompts = [REAL_PROMPT] + NOT_REAL_PROMPTS
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    feats = feats.cpu().numpy()
    return feats[0], feats[1:]


def real_photo_probability(image_embedding: np.ndarray, real_emb: np.ndarray, not_real_embs: np.ndarray, logit_scale: float) -> float:
    """P(this is a real truck photo) vs. its best-matching 'not real' prompt."""
    sims = np.concatenate([[image_embedding @ real_emb], image_embedding @ not_real_embs.T])
    logits = logit_scale * sims
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return float(probs[0])


def filter_real_photos(
    embeddings: np.ndarray,
    real_emb: np.ndarray,
    not_real_embs: np.ndarray,
    logit_scale: float,
    threshold: float = DEFAULT_REAL_PHOTO_THRESHOLD,
) -> np.ndarray:
    """Given an (N, D) array of image embeddings, returns a boolean mask of which are real truck photos."""
    return np.array(
        [real_photo_probability(e, real_emb, not_real_embs, logit_scale) >= threshold for e in embeddings]
    )


def gate_mask(embeddings: np.ndarray, gate: PhotoGate) -> np.ndarray:
    if len(embeddings) == 0:
        return np.zeros(0, dtype=bool)
    return filter_real_photos(embeddings, gate.real_emb, gate.not_real_embs, gate.logit_scale, gate.threshold)
