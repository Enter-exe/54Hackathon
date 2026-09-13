"""Adversarial validation: does the appraisal pipeline actually survive
contact with photos it's never seen? (task 10 from the original plan)

Runs six cases through the exact same ui.appraisal.run_appraisal() the demo
UI calls -- not a separate reimplementation -- against the real, currently
trained artifacts:

  1. held_out_truck  -- a real US test-split truck, never trained on
  2. dark             -- that same truck's photo, artificially darkened
  3. blurry           -- that same truck's photo, artificially blurred
  4. motorcycle       -- a real photo of something that isn't a truck
  5. single_photo     -- only one photo of a real truck (tests confidence widening)
  6. european_truck   -- a real European semi-tractor, outside the US-market
                         training distribution entirely

Results are written to artifacts/validation_report.json and printed as a
table. Results are recorded exactly as observed -- a case failing is a
finding to document, not a bug to quietly work around.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import glob
import random

from PIL import Image, ImageEnhance, ImageFilter

from ui.appraisal import load_resources, run_appraisal

FIXTURES_DIR = Path("tests/fixtures")
OUT_PATH = Path("artifacts/validation_report.json")


def make_dark(source: Path, destination: Path) -> None:
    img = Image.open(source).convert("RGB")
    ImageEnhance.Brightness(img).enhance(0.03).save(destination)


def make_blurry(source: Path, destination: Path) -> None:
    img = Image.open(source).convert("RGB")
    img.filter(ImageFilter.GaussianBlur(radius=25)).save(destination)


def pick_held_out_truck_photos() -> list[str]:
    import pandas as pd

    listings = pd.read_parquet("data/processed/listings_clean.parquet")
    splits = json.loads(Path("data/processed/splits.json").read_text())
    test_ids = set(splits["test"])
    candidates = listings[listings["ad_id"].isin(test_ids)]
    row = candidates.iloc[0]
    paths = sorted(glob.glob(f"data/images/{row['ad_id']}/*.webp"))
    return paths, row["ad_id"]


def summarize(result: dict) -> dict:
    gate = result["gate"]
    price = result["price"]
    return {
        "accepted": gate["accepted"],
        "confidence": gate["confidence"],
        "reasons": gate["reasons"],
        "price_low": price["price_low"] if price else None,
        "price_median": price["price_median"] if price else None,
        "price_high": price["price_high"] if price else None,
        "predicted_make": result["predicted_make"][0]["make_name"] if result.get("predicted_make") else None,
        "n_condition_flags": sum(1 for v in result["condition"].values() if v["flag"]) if result["condition"] else None,
    }


def validate_pipeline(tmp_dir: Path) -> dict:
    resources = load_resources()
    if resources is None:
        raise RuntimeError("Trained price model not found -- run the pipeline scripts first.")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    truck_paths, ad_id = pick_held_out_truck_photos()
    print(f"Using held-out test truck ad_id={ad_id} ({len(truck_paths)} photos)")

    report["held_out_truck"] = summarize(run_appraisal(truck_paths, resources))

    dark_path = tmp_dir / "dark.jpg"
    make_dark(truck_paths[0], dark_path)
    report["dark"] = summarize(run_appraisal([str(dark_path)], resources))

    blurry_path = tmp_dir / "blurry.jpg"
    make_blurry(truck_paths[0], blurry_path)
    report["blurry"] = summarize(run_appraisal([str(blurry_path)], resources))

    report["motorcycle"] = summarize(run_appraisal([str(FIXTURES_DIR / "motorcycle.jpg")], resources))

    report["single_photo"] = summarize(run_appraisal([truck_paths[0]], resources))

    report["european_truck"] = summarize(run_appraisal([str(FIXTURES_DIR / "european_truck.jpg")], resources))

    return report


def main():
    random.seed(0)
    tmp_dir = Path("artifacts/validation_tmp")
    report = validate_pipeline(tmp_dir)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))

    print(f"\n{'case':18s} {'accepted':10s} {'confidence':10s} {'median price':>14s}  reasons")
    for case, r in report.items():
        price = f"${r['price_median']:,.0f}" if r["price_median"] is not None else "-"
        print(f"{case:18s} {str(r['accepted']):10s} {r['confidence']:10s} {price:>14s}  {r['reasons']}")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
