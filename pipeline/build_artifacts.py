"""Build every processed-data and price-model artifact from repository inputs."""

import argparse
import importlib
import os
from pathlib import Path


EXPECTED = {
    "listings": Path("data/processed/listings_clean.csv"),
    "splits": Path("data/processed/splits.json"),
    "embeddings": Path("data/processed/listing_embeddings.npz"),
    "comparables": Path("data/processed/comparables_index.npz"),
    "condition_tags": Path("data/processed/condition_tags.csv"),
    "condition_calibration": Path("data/processed/condition_calibration.json"),
    "price_models": Path("artifacts/price_model/price_models.joblib"),
    "price_metrics": Path("artifacts/price_model/price_metrics.json"),
}


def build_artifacts(repo_root: Path, n_estimators: int = 200) -> dict[str, Path]:
    """Run the full artifact pipeline from ``repo_root`` and return its outputs."""
    repo_root = Path(repo_root).resolve()
    previous_directory = Path.cwd()
    try:
        os.chdir(repo_root)
        prepare_dataset = importlib.import_module("pipeline.prepare_dataset")
        extract_embeddings = importlib.import_module("pipeline.extract_embeddings")
        condition_assessment = importlib.import_module("pipeline.condition_assessment")
        train_price = importlib.import_module("modeling.train_price")
        data_dir = Path("data")
        processed_dir = data_dir / "processed"
        prepare_dataset.prepare_sales_data(
            data_dir / "commercial_truck_sales_100", processed_dir, Path(".")
        )
        extract_embeddings.run(data_dir)
        condition_assessment.run(data_dir)
        train_price.run_training(
            processed_dir / "listing_embeddings.npz",
            processed_dir / "listings_clean.csv",
            processed_dir / "splits.json",
            Path("artifacts/price_model"),
            n_estimators=n_estimators,
        )
        outputs = {name: repo_root / path for name, path in EXPECTED.items()}
        missing = [name for name, path in outputs.items() if not path.exists()]
        if missing:
            raise RuntimeError(f"artifact build incomplete: {missing}")
        return outputs
    finally:
        os.chdir(previous_directory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--n-estimators", type=int, default=200)
    args = parser.parse_args()

    outputs = build_artifacts(args.repo_root, args.n_estimators)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
