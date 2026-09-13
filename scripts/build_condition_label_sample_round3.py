"""Round 3 of hand-labeled condition ground truth.

Rounds 1-2 established false-positive rate (round 1, at the fixed p98
threshold: clean) and found a few genuine defects by hunting the model's
own extreme scores (round 2). But tire_wear and interior_wear stayed
mostly unevaluable in both rounds -- most listings simply don't have a
tire close-up or cab-interior photo, so random/score-based sampling wasted
most of those two columns on nulls.

This round uses find_photo_types.py's CLIP-based image-type scan
(artifacts/photo_type_scan.json) to specifically pick listings that DO
have a detected tire close-up or cab-interior photo, so labeling effort
isn't wasted. Each listing's contact sheet leads with the detected
image (so it's actually visible) plus its other photos for rust/damage
context.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from build_condition_label_sample import THUMB


def build_contact_sheet(lead_image: str, other_images: list[str], out_path: Path) -> int:
    ordered = [lead_image] + [p for p in other_images if p != lead_image]
    imgs = []
    for p in ordered[:4]:
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
    ap.add_argument("--scan", default="artifacts/photo_type_scan.json")
    ap.add_argument("--out-dir", default="artifacts/condition_labeling_round3")
    ap.add_argument("--top-k-interior", type=int, default=20)
    ap.add_argument("--top-k-tire", type=int, default=15)
    args = ap.parse_args()

    processed_dir = Path(args.data_dir) / "processed"
    out_dir = Path(args.out_dir)

    listings = pd.read_parquet(processed_dir / "listings_clean.parquet").set_index("ad_id")
    tags = pd.read_parquet(processed_dir / "condition_tags.parquet")
    scan = pd.DataFrame(json.loads(Path(args.scan).read_text()))

    top_interior = scan.nlargest(args.top_k_interior, "cab_interior_prob")
    top_tire = scan.nlargest(args.top_k_tire, "tire_closeup_prob")

    candidates = {}
    for _, r in top_interior.iterrows():
        candidates.setdefault(r["ad_id"], {})["interior_image"] = r["cab_interior_image"]
    for _, r in top_tire.iterrows():
        candidates.setdefault(r["ad_id"], {})["tire_image"] = r["tire_closeup_image"]

    manifest = []
    for ad_id, info in candidates.items():
        if ad_id not in listings.index:
            continue
        all_images = list(listings.loc[ad_id, "image_paths"])
        lead = info.get("interior_image") or info.get("tire_image")
        sheet_path = out_dir / "contact_sheets" / f"{ad_id}.jpg"
        n_used = build_contact_sheet(lead, all_images, sheet_path)
        if n_used == 0:
            continue
        manifest.append({
            "ad_id": int(ad_id),
            "contact_sheet": str(sheet_path),
            "has_interior_candidate": "interior_image" in info,
            "has_tire_candidate": "tire_image" in info,
        })

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    answer_key = tags[tags["ad_id"].isin([m["ad_id"] for m in manifest])].to_dict("records")
    (out_dir / "answer_key.json").write_text(json.dumps(answer_key, indent=2))

    print(f"Sampled {len(manifest)} listings ({sum(m['has_interior_candidate'] for m in manifest)} with a likely "
          f"interior photo, {sum(m['has_tire_candidate'] for m in manifest)} with a likely tire close-up).")
    print(f"Wrote {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
