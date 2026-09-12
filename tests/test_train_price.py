import json
import tempfile
import unittest
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from modeling.train_price import (
    evaluate,
    load_training_data,
    predict_quantiles,
    run_training,
    train_models,
)
from pipeline.data_io import write_listings_csv


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.embeddings_path = self.root / "listing_embeddings.npz"
        self.listings_path = self.root / "listings_clean.csv"
        self.splits_path = self.root / "splits.json"

        self.ids = np.array([f"truck-{i}" for i in range(10)])
        embeddings = np.arange(30, dtype=np.float32).reshape(10, 3)
        np.savez(self.embeddings_path, ad_ids=self.ids, embeddings=embeddings)
        write_listings_csv(
            pd.DataFrame(
                {
                    "ad_id": self.ids[::-1],
                    "price": np.arange(19_000, 9_000, -1_000, dtype=float),
                    "image_paths": [[] for _ in self.ids],
                }
            ),
            self.listings_path,
        )
        self.write_splits(self.ids[:6], self.ids[6:8], self.ids[8:])

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_splits(self, train, val, test):
        self.splits_path.write_text(
            json.dumps(
                {
                    "train": [str(value) for value in train],
                    "val": [str(value) for value in val],
                    "test": [str(value) for value in test],
                }
            )
        )

    def test_load_training_data_aligns_embeddings_prices_and_splits(self):
        split_data = load_training_data(
            self.embeddings_path, self.listings_path, self.splits_path
        )

        self.assertEqual(split_data["train"][0].shape, (6, 3))
        np.testing.assert_array_equal(
            split_data["train"][1],
            [10_000, 11_000, 12_000, 13_000, 14_000, 15_000],
        )

    def test_load_training_data_rejects_overlapping_splits(self):
        self.write_splits(self.ids[:6], self.ids[5:8], self.ids[8:])

        with self.assertRaisesRegex(ValueError, "overlap"):
            load_training_data(
                self.embeddings_path, self.listings_path, self.splits_path
            )

    def test_run_training_writes_model_bundle_and_metrics(self):
        output_dir = self.root / "artifacts"

        metrics = run_training(
            self.embeddings_path,
            self.listings_path,
            self.splits_path,
            output_dir,
            n_estimators=20,
        )

        bundle = joblib.load(output_dir / "price_models.joblib")
        saved_metrics = json.loads((output_dir / "price_metrics.json").read_text())
        self.assertEqual(bundle["quantiles"], [0.1, 0.5, 0.9])
        self.assertEqual(set(bundle["models"]), {0.1, 0.5, 0.9})
        self.assertEqual(saved_metrics, metrics)
        self.assertEqual(set(metrics), {"val", "test"})


class PriceModelTests(unittest.TestCase):
    def test_quantile_models_produce_ordered_predictions_and_metrics(self):
        generator = np.random.default_rng(0)
        X = generator.normal(size=(60, 4))
        y = 20_000 + 4_000 * X[:, 0] - 2_000 * X[:, 1]

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            models = train_models(X[:50], y[:50], n_estimators=20)
        predictions = predict_quantiles(models, X[50:])
        metrics = evaluate(models, X[50:], y[50:])

        self.assertEqual(predictions.shape, (10, 3))
        self.assertTrue(np.all(predictions[:, 0] <= predictions[:, 1]))
        self.assertTrue(np.all(predictions[:, 1] <= predictions[:, 2]))
        self.assertEqual(
            set(metrics),
            {
                "pinball_q10",
                "pinball_q50",
                "pinball_q90",
                "median_mae",
                "median_r2",
                "interval_coverage",
            },
        )
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
