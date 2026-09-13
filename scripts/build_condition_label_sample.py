"""Build a sample of listings for hand-labeling ground-truth condition, to
evaluate pipeline/condition_assessment.py's zero-shot flags against real
judgment instead of just checking the code runs.

Sampled from val+test only (never train, which calibrated the thresholds),
stratified half from listings flagged on >=1 attribute and half from
listings flagged on none, so the label set has enough positives per
attribute to compute recall from -- a pure random sample at the ~20% base
flag rate would be positive-starved.

Each sampled listing's photos are combined into a single contact-sheet
image (not shown alongside model scores) so a human -- or an
image-viewing labeler -- can judge it fast, without the predicted
scores biasing the judgment. The answer key (scores/flags) is written
separately for eval_condition_assessment.py to compare against later.
"""
import argparse
import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image

THUMB = 320


def build_contact_sheet(image_paths: list[str], out_path: Path) -> int:
    imgs = []
    for p in image_paths[:4]:
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((THUMB, THUMB))
            imgs.append(im)
        except Exception:
            continue
    if not imgs:
        return 0

    cols = 2
    rows = (len(imgs) + 1) // 2
    sheet = Image.new("RGB", (cols * THUMB, rows * THUMB), (30, 30, 30))
    for i, im in enumerate(imgs):
        x, y = (i % cols) * THUMB, (i // cols) * THUMB
        sheet.paste(im, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, format="JPEG", quality=85)
    return len(imgs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="artifacts/condition_labeling")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    processed_dir = Path(args.data_dir) / "processed"
    out_dir = Path(args.out_dir)

    listings = pd.read_parquet(processed_dir / "listings_clean.parquet")
    tags = pd.read_parquet(processed_dir / "condition_tags.parquet")

    held_out = tags[tags["split"].isin(["val", "test"])].copy()
    flag_cols = [c for c in held_out.columns if c.endswith("_flag")]
    held_out["any_flag"] = held_out[flag_cols].any(axis=1)

    rng = random.Random(args.seed)
    flagged_ids = held_out.loc[held_out["any_flag"], "ad_id"].tolist()
    clean_ids = held_out.loc[~held_out["any_flag"], "ad_id"].tolist()
    rng.shuffle(flagged_ids)
    rng.shuffle(clean_ids)

    half = args.n // 2
    sample_ids = flagged_ids[:half] + clean_ids[: args.n - min(half, len(flagged_ids))]
    rng.shuffle(sample_ids)

    listings_by_id = listings.set_index("ad_id")
    manifest = []
    for ad_id in sample_ids:
        if ad_id not in listings_by_id.index:
            continue
        image_paths = list(listings_by_id.loc[ad_id, "image_paths"])
        sheet_path = out_dir / "contact_sheets" / f"{ad_id}.jpg"
        n_used = build_contact_sheet(image_paths, sheet_path)
        if n_used == 0:
            continue
        manifest.append({"ad_id": int(ad_id), "contact_sheet": str(sheet_path), "n_images": n_used})

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    answer_key = tags[tags["ad_id"].isin([m["ad_id"] for m in manifest])].to_dict("records")
    (out_dir / "answer_key.json").write_text(json.dumps(answer_key, indent=2))

    print(f"Sampled {len(manifest)} listings ({len(flagged_ids[:half])} flagged, "
          f"{len(manifest) - len(flagged_ids[:half])} clean) from val+test (n={len(held_out)}).")
    print(f"Wrote {out_dir / 'manifest.json'} (label these, WITHOUT peeking at answer_key.json)")
    print(f"Wrote {out_dir / 'answer_key.json'} (predicted scores, for eval only)")


if __name__ == "__main__":
    main()
