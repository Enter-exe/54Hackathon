"""Zero-shot condition assessment from CLIP embeddings -- no labeled
condition data, no training. This is what replaces the manual
condition-score labeling in the original spec.

For each condition attribute (rust, body damage, tire wear, interior wear),
two text prompts are defined: one describing the problem, one describing
the OK state. An image's score for that attribute is P(problem) from a
softmax over the image's cosine similarity to the two prompts (the
standard CLIP zero-shot classification trick -- comparing a paired
problem/ok prompt is far less noisy than thresholding one prompt's raw
cosine similarity in isolation).

A listing's score per attribute is the MAX over its images, not the mean:
a dent visible in only one of four photos should still flag the listing,
not get diluted by three clean angles.

IMPORTANT calibration note: a flat 0.5 cutoff on P(problem) does NOT mean
"more likely damaged than not". On the real dataset, "rust" is well-behaved
(near-zero for most listings, a small genuine high tail), but "body_damage"
and "interior_wear" come out systematically high (medians ~0.6-0.65) across
nearly the whole fleet, including obviously new trucks -- CLIP's softmax
over a problem/ok prompt pair isn't automatically balanced at 0.5; one
phrasing can just win by default regardless of image content. So instead
of an absolute threshold, each attribute is calibrated relative to the
TRAIN split's own score distribution (e.g. "worse than 80% of comparable
listings"), and that per-attribute threshold is reused at inference.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch

from pipeline.extract_embeddings import MODEL_NAME, embed_images, load_model
from pipeline.gating import build_gate, gate_mask

ATTRIBUTES = [
    {
        "name": "rust",
        "problem_prompt": "a photo of a rusty, corroded truck exterior",
        "ok_prompt": "a photo of a clean truck exterior with no rust",
    },
    {
        "name": "body_damage",
        "problem_prompt": "a photo of a truck with dents or damaged body panels",
        "ok_prompt": "a photo of a truck with smooth, undamaged body panels",
    },
    {
        "name": "tire_wear",
        "problem_prompt": "a photo of worn, cracked, or balding truck tires",
        "ok_prompt": "a photo of truck tires in good condition",
    },
    {
        "name": "interior_wear",
        "problem_prompt": "a photo of a truck cab interior showing wear, damage, or heavy dirt",
        "ok_prompt": "a photo of a clean, well-maintained truck cab interior",
    },
]

DEFAULT_PERCENTILE = 80.0  # flag a listing if it scores worse than this percentile of the train-split fleet


def load_text_embeddings(model, device: str) -> dict:
    """Returns {attribute_name: (problem_embedding, ok_embedding)}, both L2-normalized."""
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    prompts = []
    for attr in ATTRIBUTES:
        prompts += [attr["problem_prompt"], attr["ok_prompt"]]
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        text_feats = model.encode_text(tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    text_feats = text_feats.cpu().numpy()
    return {
        attr["name"]: (text_feats[2 * i], text_feats[2 * i + 1])
        for i, attr in enumerate(ATTRIBUTES)
    }


def score_embedding(image_embedding: np.ndarray, text_pairs: dict, logit_scale: float) -> dict:
    """One image's P(problem) per attribute."""
    scores = {}
    for name, (problem_emb, ok_emb) in text_pairs.items():
        logits = logit_scale * np.array(
            [float(image_embedding @ problem_emb), float(image_embedding @ ok_emb)]
        )
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        scores[name] = float(probs[0])
    return scores


def assess_images(
    image_paths: list[str],
    model,
    preprocess,
    device: str,
    text_pairs: dict,
    thresholds: dict | None = None,
    gate=None,
) -> dict | None:
    """Assess a listing's condition from its image paths.

    Returns {attribute_name: {"probability": float, "flag": bool}} when
    `thresholds` (per-attribute cutoffs from calibrate_thresholds) is given,
    otherwise {attribute_name: {"probability": float}}. Returns None if none
    of the given paths were usable images.

    gate: an optional pipeline.gating.PhotoGate. When given, images that
    aren't actually real truck photos (e.g. a dealer "Photos Coming Soon"
    placeholder graphic) are excluded before scoring -- one of those was
    found scoring as a top "rust" hit before this filter was added.
    """
    valid_paths = [p for p in image_paths if Path(p).exists()]
    if not valid_paths:
        return None
    embs = embed_images(valid_paths, model, preprocess, device)
    keep_mask = np.any(embs != 0, axis=1)
    if gate is not None:
        keep_mask &= gate_mask(embs, gate)
    valid_embs = embs[keep_mask]
    if len(valid_embs) == 0:
        return None

    logit_scale = model.logit_scale.exp().item()
    per_image_scores = [score_embedding(e, text_pairs, logit_scale) for e in valid_embs]

    result = {}
    for attr in ATTRIBUTES:
        name = attr["name"]
        max_prob = max(s[name] for s in per_image_scores)
        entry = {"probability": max_prob}
        if thresholds is not None:
            entry["flag"] = max_prob >= thresholds[name]
        result[name] = entry
    return result


def calibrate_thresholds(prob_by_attr: dict, percentile: float = DEFAULT_PERCENTILE) -> dict:
    """Per-attribute threshold = the given percentile of a reference (train-split) score
    distribution, so a listing is flagged relative to the fleet rather than against an
    absolute cutoff CLIP's raw softmax was never guaranteed to be calibrated against."""
    return {name: float(np.percentile(probs, percentile)) for name, probs in prob_by_attr.items()}


def run(data_dir: Path, percentile: float = DEFAULT_PERCENTILE) -> None:
    processed_dir = data_dir / "processed"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    df = pd.read_parquet(processed_dir / "listings_clean.parquet")
    with open(processed_dir / "splits.json") as f:
        splits = json.load(f)
    split_of = {ad_id: split_name for split_name, ids in splits.items() for ad_id in ids}
    df.loc[:, "split"] = df["ad_id"].map(split_of)

    print(f"Loading {MODEL_NAME}...")
    model, preprocess = load_model(device)
    text_pairs = load_text_embeddings(model, device)
    gate = build_gate(model, device, MODEL_NAME)

    rows = []
    n = len(df)
    for i, row in enumerate(df.itertuples(), 1):
        assessment = assess_images(row.image_paths, model, preprocess, device, text_pairs, gate=gate)
        if assessment is None:
            print(f"WARN: no usable (real-photo) images for ad_id={row.ad_id}, skipping")
            continue
        out_row = {"ad_id": row.ad_id, "split": row.split}
        for name, v in assessment.items():
            out_row[f"{name}_prob"] = v["probability"]
        rows.append(out_row)
        if i % 100 == 0 or i == n:
            print(f"  assessed {i}/{n} listings")

    out_df = pd.DataFrame(rows)
    train_mask = out_df["split"] == "train"
    thresholds = calibrate_thresholds(
        {attr["name"]: out_df.loc[train_mask, f"{attr['name']}_prob"].to_numpy() for attr in ATTRIBUTES},
        percentile=percentile,
    )
    for attr in ATTRIBUTES:
        name = attr["name"]
        out_df.loc[:, f"{name}_flag"] = out_df[f"{name}_prob"] >= thresholds[name]

    out_df.to_parquet(processed_dir / "condition_tags.parquet", index=False)
    with open(processed_dir / "condition_calibration.json", "w") as f:
        json.dump({"percentile": percentile, "thresholds": thresholds}, f, indent=2)
    print(f"Wrote {processed_dir / 'condition_tags.parquet'} ({len(out_df)} listings)")
    print(f"Wrote {processed_dir / 'condition_calibration.json'}")
    for attr in ATTRIBUTES:
        name = attr["name"]
        flagged = out_df[f"{name}_flag"].sum()
        print(f"  {name}: threshold={thresholds[name]:.3f} (p{percentile} of train fleet), {flagged}/{len(out_df)} flagged")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE)
    args = ap.parse_args()
    run(Path(args.data_dir), args.percentile)


if __name__ == "__main__":
    main()
