"""Find which listings actually have a tire close-up or cab-interior photo,
across the WHOLE dataset, using CLIP zero-shot image classification (the
same problem/ok-prompt-pair trick pipeline/condition_assessment.py uses for
condition scoring, applied here as an image-type detector instead).

Motivation: hand-labeling ground truth for tire_wear/interior_wear has been
badly sample-inefficient so far -- most CTT listings only photograph the
truck's front-3/4, front, side, and rear, so most contact sheets picked by
random or by-score sampling had NO tire close-up or cab-interior shot to
judge from at all (see artifacts/condition_labeling(_round2)/labels.json --
most interior_wear/tire_wear entries are null for exactly this reason).
This scans every image of every not-yet-labeled listing and ranks listings
by "most likely to actually contain a tire close-up" / "cab interior",
so the next labeling round's images are usable instead of mostly wasted.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.extract_embeddings import MODEL_NAME, load_model

PROMPT_PAIRS = {
    "tire_closeup": (
        "a close-up photo of a single truck tire and wheel",
        "a wide photo of an entire truck",
    ),
    "cab_interior": (
        "a photo of the inside of a truck cab, showing the dashboard, steering wheel, and seats",
        "a photo of the outside of a truck",
    ),
}


def score_images(image_paths, model, preprocess, device, text_pairs, logit_scale) -> dict:
    """Returns {prompt_name: (best_prob, best_image_path)} across this listing's images."""
    result = {name: (0.0, None) for name in text_pairs}
    for p in image_paths:
        if not Path(p).exists():
            continue
        try:
            img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        except Exception:
            continue
        with torch.no_grad():
            feat = model.encode_image(img)
            feat = (feat / feat.norm(dim=-1, keepdim=True)).cpu().numpy()[0]
        for name, (pos_emb, neg_emb) in text_pairs.items():
            logits = logit_scale * np.array([float(feat @ pos_emb), float(feat @ neg_emb)])
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            prob = float(probs[0])
            if prob > result[name][0]:
                result[name] = (prob, p)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--exclude-labels", nargs="*", default=[
        "artifacts/condition_labeling/labels.json",
        "artifacts/condition_labeling_round2/labels.json",
    ])
    ap.add_argument("--out", default="artifacts/photo_type_scan.json")
    ap.add_argument("--limit", type=int, default=None, help="cap listings scanned, for a quick test run")
    args = ap.parse_args()

    processed_dir = Path(args.data_dir) / "processed"
    listings = pd.read_parquet(processed_dir / "listings_clean.parquet")

    already_labeled = set()
    for p in args.exclude_labels:
        if Path(p).exists():
            already_labeled |= set(json.loads(Path(p).read_text()).keys())
    listings = listings[~listings["ad_id"].astype(str).isin(already_labeled)]
    if args.limit:
        listings = listings.head(args.limit)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}, scanning {len(listings)} listings (excluding {len(already_labeled)} already labeled)")
    model, preprocess = load_model(device)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    prompts = [p for pair in PROMPT_PAIRS.values() for p in pair]
    with torch.no_grad():
        text_feats = model.encode_text(tokenizer(prompts).to(device))
        text_feats = (text_feats / text_feats.norm(dim=-1, keepdim=True)).cpu().numpy()
    text_pairs = {name: (text_feats[2 * i], text_feats[2 * i + 1]) for i, name in enumerate(PROMPT_PAIRS)}
    logit_scale = model.logit_scale.exp().item()

    rows = []
    n = len(listings)
    for i, row in enumerate(listings.itertuples(), 1):
        scores = score_images(list(row.image_paths), model, preprocess, device, text_pairs, logit_scale)
        rows.append({
            "ad_id": int(row.ad_id),
            "tire_closeup_prob": scores["tire_closeup"][0],
            "tire_closeup_image": scores["tire_closeup"][1],
            "cab_interior_prob": scores["cab_interior"][0],
            "cab_interior_image": scores["cab_interior"][1],
        })
        if i % 100 == 0 or i == n:
            print(f"  scanned {i}/{n}")

    Path(args.out).write_text(json.dumps(rows, indent=2))
    df = pd.DataFrame(rows)
    print(f"\nWrote {args.out}")
    print(f"Likely tire close-ups (prob>0.9): {(df['tire_closeup_prob'] > 0.9).sum()}")
    print(f"Likely cab interiors (prob>0.9): {(df['cab_interior_prob'] > 0.9).sum()}")


if __name__ == "__main__":
    main()
