"""Run deterministic end-to-end and adversarial appraisal checks."""

import argparse
import json
import tempfile
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

from pipeline.appraise import AppraisalEngine
from pipeline.build_artifacts import EXPECTED
from pipeline.data_io import read_listings_csv


def make_dark(source: Path, destination: Path) -> None:
    """Write a deterministic near-black copy of ``source``."""
    with Image.open(source) as image:
        ImageEnhance.Brightness(image).enhance(0.03).save(destination)


def make_blurry(source: Path, destination: Path) -> None:
    """Write a deterministic heavily blurred copy of ``source``."""
    with Image.open(source) as image:
        image.filter(ImageFilter.GaussianBlur(radius=25)).save(destination)


def _summary(result: dict) -> dict:
    return {
        "accepted": bool(result["accepted"]),
        "confidence": str(result["confidence"]),
        "reasons": [str(reason) for reason in result["reasons"]],
        **{
            name: None if result[name] is None else float(result[name])
            for name in ("price_low", "price_median", "price_high")
        },
        "condition_flags": {
            str(name): bool(details["flag"])
            for name, details in result["condition"].items()
        },
        "comparable_ids": [str(item["ad_id"]) for item in result["comparables"]],
    }


def _absolute_paths(repo_root: Path, paths: list[str]) -> list[str]:
    return [
        str(path if path.is_absolute() else repo_root / path)
        for path in map(Path, paths)
    ]


def validate_pipeline(
    repo_root: Path, engine: AppraisalEngine | None = None
) -> dict:
    """Run six appraisal cases and write their compact result summary."""
    repo_root = Path(repo_root).resolve()
    listings = read_listings_csv(repo_root / EXPECTED["listings"])
    splits = json.loads((repo_root / EXPECTED["splits"]).read_text())
    test_id = str(splits["test"][0])
    held_out = listings.loc[listings["ad_id"].astype(str) == test_id]
    if held_out.empty:
        raise ValueError(f"test listing {test_id!r} is missing from listings")
    truck_paths = _absolute_paths(repo_root, held_out.iloc[0]["image_paths"])
    if not truck_paths:
        raise ValueError(f"test listing {test_id!r} has no images")

    appraisal_engine = engine or AppraisalEngine.from_artifacts(repo_root)
    fixture_dir = repo_root / "tests/fixtures"
    with tempfile.TemporaryDirectory(prefix="kamion-validation-") as directory:
        dark_path = Path(directory) / "dark.jpg"
        blurry_path = Path(directory) / "blurry.jpg"
        make_dark(Path(truck_paths[0]), dark_path)
        make_blurry(Path(truck_paths[0]), blurry_path)
        cases = {
            "held_out_truck": truck_paths,
            "dark": [str(dark_path)],
            "blurry": [str(blurry_path)],
            "motorcycle": [str(fixture_dir / "motorcycle.jpg")],
            "single_photo": truck_paths[:1],
            "non_us_truck": [str(fixture_dir / "european_truck.jpg")],
        }
        report = {
            name: _summary(appraisal_engine.appraise(paths))
            for name, paths in cases.items()
        }

    report_path = repo_root / "artifacts/validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps(validate_pipeline(args.repo_root), indent=2))


if __name__ == "__main__":
    main()
