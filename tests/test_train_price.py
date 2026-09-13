import json
import tempfile
import unittest
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from modeling.train_price import (
    compute_calibration_bounds,
    evaluate,
    load_training_data,
    predict_range,
    run_training,
    train_median_model,
)


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.embeddings_path = self.root / "listing_embeddings.npz"
        self.listings_path = self.root / "listings_clean.parquet"
        self.splits_path = self.root / "splits.json"

        self.ids = np.array([f"truck-{i}" for i in range(10)])
        embeddings = np.arange(30, dtype=np.float32).reshape(10, 3)
        np.savez(self.embeddings_path, ad_ids=self.ids, embeddings=embeddings)
        pd.DataFrame(
            {
                "ad_id": self.ids[::-1],
                "price": np.arange(19_000, 9_000, -1_000, dtype=float),
            }
        ).to_parquet(self.listings_path, index=False)
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

    def write_condition_tags(self, ad_ids):
        path = self.root / "condition_tags.parquet"
        rng = np.random.default_rng(0)
        pd.DataFrame(
            {
                "ad_id": list(ad_ids),
                "rust_prob": rng.uniform(size=len(ad_ids)),
                "body_damage_prob": rng.uniform(size=len(ad_ids)),
                "tire_wear_prob": rng.uniform(size=len(ad_ids)),
                "interior_wear_prob": rng.uniform(size=len(ad_ids)),
            }
        ).to_parquet(path, index=False)
        return path

    def test_load_training_data_concatenates_condition_features_when_given(self):
        condition_path = self.write_condition_tags(self.ids)

        split_data = load_training_data(
            self.embeddings_path, self.listings_path, self.splits_path, condition_path
        )

        self.assertEqual(split_data["train"][0].shape, (6, 3 + 4))

    def test_load_training_data_skips_condition_tags_file_that_does_not_exist(self):
        missing_path = self.root / "does_not_exist.parquet"

        split_data = load_training_data(
            self.embeddings_path, self.listings_path, self.splits_path, missing_path
        )

        self.assertEqual(split_data["train"][0].shape, (6, 3))

    def test_load_training_data_raises_on_incomplete_condition_tags(self):
        condition_path = self.write_condition_tags(self.ids[:-1])  # missing one ad_id

        with self.assertRaisesRegex(ValueError, "missing condition tags"):
            load_training_data(
                self.embeddings_path, self.listings_path, self.splits_path, condition_path
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
        self.assertIn("model", bundle)
        # sign isn't guaranteed with only 2 calibration points (this fixture's val
        # split) -- see CalibrationTests below for the sign check with real data.
        self.assertTrue(np.isfinite(bundle["rel_lo"]))
        self.assertTrue(np.isfinite(bundle["rel_hi"]))
        self.assertEqual(saved_metrics, metrics)
        self.assertEqual(set(metrics), {"val", "test", "rel_lo", "rel_hi"})

    def test_run_training_accepts_condition_tags(self):
        output_dir = self.root / "artifacts"
        condition_path = self.write_condition_tags(self.ids)

        metrics = run_training(
            self.embeddings_path,
            self.listings_path,
            self.splits_path,
            output_dir,
            n_estimators=20,
            condition_tags_path=condition_path,
        )

        self.assertEqual(set(metrics), {"val", "test", "rel_lo", "rel_hi"})


class PriceModelTests(unittest.TestCase):
    def test_median_model_produces_ordered_range_predictions_and_metrics(self):
        generator = np.random.default_rng(0)
        X = generator.normal(size=(60, 4))
        y = 20_000 + 4_000 * X[:, 0] - 2_000 * X[:, 1]
        y = np.abs(y) + 5_000  # keep strictly positive: relative bounds require it

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = train_median_model(X[:50], y[:50], n_estimators=20)
        predictions = predict_range(model, X[50:], rel_lo=-0.2, rel_hi=0.3)
        metrics = evaluate(model, X[50:], y[50:], rel_lo=-0.2, rel_hi=0.3)

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

    def test_predict_range_scales_with_the_prediction_not_a_fixed_dollar_amount(self):
        """The whole point of the relative-bounds redesign: a truck predicted
        at $100k should get a much wider dollar range than one predicted at
        $10k, for the same rel_lo/rel_hi -- unlike a fixed additive margin."""
        generator = np.random.default_rng(3)
        X = generator.normal(size=(60, 3))
        y = 10_000 + 500 * np.abs(X[:, 0]) + 200 * np.arange(60)  # wide, positive, spread out
        model = train_median_model(X, y, n_estimators=20)

        cheap_row = X[np.argmin(model.predict(X))].reshape(1, -1)
        expensive_row = X[np.argmax(model.predict(X))].reshape(1, -1)

        cheap_pred = predict_range(model, cheap_row, rel_lo=-0.3, rel_hi=0.3)[0]
        expensive_pred = predict_range(model, expensive_row, rel_lo=-0.3, rel_hi=0.3)[0]

        cheap_width = cheap_pred[2] - cheap_pred[0]
        expensive_width = expensive_pred[2] - expensive_pred[0]
        self.assertGreater(expensive_width, cheap_width)


class CalibrationTests(unittest.TestCase):
    def test_compute_calibration_bounds_returns_finite_floats_straddling_zero(self):
        generator = np.random.default_rng(1)
        X = generator.normal(size=(80, 4))
        y = 20_000 + 4_000 * X[:, 0]
        y = np.abs(y) + 5_000
        model = train_median_model(X[:60], y[:60], n_estimators=20)

        rel_lo, rel_hi = compute_calibration_bounds(model, X[60:], y[60:], target_coverage=0.8)

        self.assertTrue(np.isfinite(rel_lo))
        self.assertTrue(np.isfinite(rel_hi))
        self.assertLessEqual(rel_lo, 0.0)
        self.assertGreaterEqual(rel_hi, 0.0)

    def test_calibration_bounds_raise_calibration_set_coverage_to_target(self):
        """The whole point of conformal calibration: applying the bounds to
        the SAME data they were computed from must hit at least the target
        coverage, by construction."""
        generator = np.random.default_rng(2)
        X = generator.normal(size=(100, 4))
        y = 20_000 + 4_000 * X[:, 0] - 1_500 * X[:, 1] ** 2  # nonlinear -> raw predictions undercover
        y = np.abs(y) + 5_000
        model = train_median_model(X[:70], y[:70], n_estimators=20)
        X_cal, y_cal = X[70:], y[70:]

        raw_coverage = evaluate(model, X_cal, y_cal, rel_lo=0.0, rel_hi=0.0)["interval_coverage"]
        rel_lo, rel_hi = compute_calibration_bounds(model, X_cal, y_cal, target_coverage=0.8)
        calibrated_coverage = evaluate(model, X_cal, y_cal, rel_lo=rel_lo, rel_hi=rel_hi)["interval_coverage"]

        self.assertGreaterEqual(calibrated_coverage, 0.8)
        self.assertGreaterEqual(calibrated_coverage, raw_coverage)


if __name__ == "__main__":
    unittest.main()
