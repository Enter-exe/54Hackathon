"""Extract frozen CLIP embeddings for every listing and build the
train-set similarity index used for "comparable listings" at inference.

For each listing: encode each of its (up to 4) images with CLIP ViT-B/32,
mean-pool into one embedding vector. No fine-tuning -- this is a frozen,
off-the-shelf encoder, so there's no leakage path between this step and
the price model trained on its output (see pipeline/clean_split.py docstring
for why that means we don't need a held-out split just for this).

Output:
  data/processed/embeddings.parquet   -- ad_id, split, price, embedding (list[float])
  data/processed/comparables_index.npz -- train-only embeddings + metadata,
                                           for nearest-neighbor lookup at inference
"""
import argparse
import json
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image

MODEL_NAME = "ViT-B-32-quickgelu"  # matches OpenAI's original activation; plain ViT-B-32 mismatches quick_gelu and silently degrades embeddings
PRETRAINED = "openai"


def load_model(device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(device).eval()
    return model, preprocess


@torch.no_grad()
def embed_images(paths: list[str], model, preprocess, device: str, batch_size: int = 32) -> np.ndarray:
    """Encode a flat list of image paths, return one embedding per image."""
    embs = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        imgs = []
        keep_idx = []
        for j, p in enumerate(batch_paths):
            try:
                imgs.append(preprocess(Image.open(p).convert("RGB")))
                keep_idx.append(j)
            except Exception as e:
                print(f"WARN: could not load {p}: {e}")
        if not imgs:
            continue
        batch = torch.stack(imgs).to(device)
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.cpu().numpy()
        out = np.zeros((len(batch_paths), feats.shape[1]), dtype=np.float32)
        for k, j in enumerate(keep_idx):
            out[j] = feats[k]
        embs.append(out)
    return np.concatenate(embs, axis=0) if embs else np.zeros((0, 512), dtype=np.float32)


def pooled_listing_embedding(image_paths: list[str], model, preprocess, device: str) -> np.ndarray | None:
    valid_paths = [p for p in image_paths if Path(p).exists()]
    if not valid_paths:
        return None
    embs = embed_images(valid_paths, model, preprocess, device)
    nonzero = embs[np.any(embs != 0, axis=1)]
    if len(nonzero) == 0:
        return None
    pooled = nonzero.mean(axis=0)
    pooled = pooled / (np.linalg.norm(pooled) + 1e-8)
    return pooled.astype(np.float32)


def run(data_dir: Path) -> None:
    processed_dir = data_dir / "processed"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    df = pd.read_parquet(processed_dir / "listings_clean.parquet")
    with open(processed_dir / "splits.json") as f:
        splits = json.load(f)
    split_of = {}
    for split_name, ids in splits.items():
        for ad_id in ids:
            split_of[ad_id] = split_name
    df.loc[:, "split"] = df["ad_id"].map(split_of)

    print(f"Loading {MODEL_NAME} ({PRETRAINED})...")
    model, preprocess = load_model(device)

    rows = []
    n = len(df)
    for i, row in enumerate(df.itertuples(), 1):
        pooled = pooled_listing_embedding(row.image_paths, model, preprocess, device)
        if pooled is None:
            print(f"WARN: no usable images for ad_id={row.ad_id}, skipping")
            continue
        rows.append(
            {
                "ad_id": row.ad_id,
                "split": row.split,
                "price": row.price,
                "year": row.year,
                "make_name": row.make_name,
                "model_name": row.model_name,
                "embedding": pooled.tolist(),
            }
        )
        if i % 100 == 0 or i == n:
            print(f"  embedded {i}/{n} listings")

    emb_df = pd.DataFrame(rows)
    emb_df.to_parquet(processed_dir / "embeddings.parquet", index=False)
    print(f"Wrote {processed_dir / 'embeddings.parquet'} ({len(emb_df)} listings)")

    train_df = emb_df[emb_df["split"] == "train"]
    train_embeddings = np.stack(train_df["embedding"].to_numpy())
    np.savez(
        processed_dir / "comparables_index.npz",
        embeddings=train_embeddings,
        ad_id=train_df["ad_id"].to_numpy(),
        price=train_df["price"].to_numpy(),
        year=train_df["year"].to_numpy(),
        make_name=train_df["make_name"].to_numpy(),
        model_name=train_df["model_name"].to_numpy(),
    )
    print(f"Wrote {processed_dir / 'comparables_index.npz'} ({len(train_df)} train listings)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    run(Path(args.data_dir))


if __name__ == "__main__":
    main()
