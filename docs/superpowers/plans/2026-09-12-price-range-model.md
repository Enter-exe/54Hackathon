# Price-Range Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train, evaluate, and save three LightGBM quantile regressors that map cached listing-level image embeddings to lower, median, and upper truck-price estimates.

**Architecture:** A single training module loads Task 3's NPZ embedding cache, joins it to Task 2's cleaned prices by `ad_id`, and applies the listing IDs in `splits.json`. It trains q10/q50/q90 models, evaluates validation and test predictions, and writes one model bundle plus a JSON metrics report.

**Tech Stack:** Python 3, NumPy, pandas, LightGBM, scikit-learn, joblib, unittest

**Spec:** `docs/main_specs.md`

## Global Constraints

- Consume `data/processed/listing_embeddings.npz` with `ad_ids` shaped `(N,)` and finite `float32` `embeddings` shaped `(N, D)`.
- Consume cleaned `ad_id` and `price` columns from `data/processed/listings_clean.parquet`.
- Consume non-overlapping `train`, `val`, and `test` listing IDs from `data/processed/splits.json`.
- Train separate LightGBM quantile regressors for q10, q50, and q90 without fine-tuning the image encoder.
- Report pinball loss for all quantiles, median MAE and R², and q10–q90 interval coverage on validation and test sets.
- Do not add tabular vehicle attributes to the model input.

---

### Task 1: Training-data contract and validation

**Files:**
- Create: `modeling/train_price.py`
- Create: `tests/test_train_price.py`

**Interfaces:**
- Consumes: NPZ arrays `ad_ids: numpy.ndarray[str]`, `embeddings: numpy.ndarray[float32]`; Parquet columns `ad_id`, `price`; JSON keys `train`, `val`, `test`.
- Produces: `load_training_data(embeddings_path, listings_path, splits_path) -> dict[str, tuple[numpy.ndarray, numpy.ndarray]]`.

- [x] **Step 1: Write the failing contract tests**

```python
def test_load_training_data_aligns_embeddings_prices_and_splits(self):
    split_data = load_training_data(self.embeddings, self.listings, self.splits)
    self.assertEqual(split_data["train"][0].shape, (6, 3))
    np.testing.assert_array_equal(split_data["train"][1], [10_000, 11_000, 12_000, 13_000, 14_000, 15_000])

def test_load_training_data_rejects_overlapping_splits(self):
    self.write_splits(train=self.ids[:6], val=self.ids[5:8], test=self.ids[8:])
    with self.assertRaisesRegex(ValueError, "overlap"):
        load_training_data(self.embeddings, self.listings, self.splits)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest tests.test_train_price -v`
Expected: FAIL because `modeling.train_price` does not exist.

- [x] **Step 3: Implement minimal validated loading**

```python
def load_training_data(embeddings_path, listings_path, splits_path):
    cache = np.load(embeddings_path, allow_pickle=False)
    ad_ids = cache["ad_ids"].astype(str)
    embeddings = cache["embeddings"]
    listings = pd.read_parquet(listings_path).set_index("ad_id")
    prices = listings.loc[ad_ids, "price"].to_numpy(dtype=float)
    splits = json.loads(Path(splits_path).read_text())
    return {
        name: (embeddings[np.isin(ad_ids, ids)], prices[np.isin(ad_ids, ids)])
        for name, ids in splits.items()
    }
```

Add explicit validation for required keys and columns, array shape, finite values, unique IDs, positive prices, split names, split overlap, and exact split coverage.

- [x] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest tests.test_train_price -v`
Expected: both data-contract tests PASS.

### Task 2: Quantile training, ordered prediction, and metrics

**Files:**
- Modify: `modeling/train_price.py`
- Modify: `tests/test_train_price.py`
- Create: `modeling/requirements.txt`

**Interfaces:**
- Consumes: split matrices returned by `load_training_data`.
- Produces: `train_models(X, y, n_estimators=200) -> dict[float, lightgbm.LGBMRegressor]`, `predict_quantiles(models, X) -> numpy.ndarray`, and `evaluate(models, X, y) -> dict[str, float]`.

- [x] **Step 1: Write failing model-behavior tests**

```python
def test_quantile_models_produce_ordered_predictions_and_metrics(self):
    models = train_models(self.X_train, self.y_train, n_estimators=20)
    predictions = predict_quantiles(models, self.X_test)
    self.assertEqual(predictions.shape, (len(self.X_test), 3))
    self.assertTrue(np.all(predictions[:, 0] <= predictions[:, 1]))
    self.assertTrue(np.all(predictions[:, 1] <= predictions[:, 2]))
    metrics = evaluate(models, self.X_test, self.y_test)
    self.assertEqual(set(metrics), {"pinball_q10", "pinball_q50", "pinball_q90", "median_mae", "median_r2", "interval_coverage"})
```

- [x] **Step 2: Run the test to verify it fails**

Run: `python3 -m unittest tests.test_train_price.PriceModelTests.test_quantile_models_produce_ordered_predictions_and_metrics -v`
Expected: FAIL because the training functions do not exist.

- [x] **Step 3: Implement the three regressors and evaluation**

```python
QUANTILES = (0.1, 0.5, 0.9)

def train_models(X, y, n_estimators=200):
    return {
        q: LGBMRegressor(objective="quantile", alpha=q, n_estimators=n_estimators, random_state=0, verbosity=-1).fit(X, y)
        for q in QUANTILES
    }

def predict_quantiles(models, X):
    return np.sort(np.column_stack([models[q].predict(X) for q in QUANTILES]), axis=1)
```

Use scikit-learn's `mean_pinball_loss`, `mean_absolute_error`, and `r2_score`; calculate inclusive interval coverage from the ordered lower and upper predictions.

- [x] **Step 4: Run the full test file**

Run: `python3 -m unittest tests.test_train_price -v`
Expected: all data and model tests PASS.

### Task 3: Command-line training and saved artifacts

**Files:**
- Modify: `modeling/train_price.py`
- Modify: `tests/test_train_price.py`

**Interfaces:**
- Consumes: paths supplied through `--embeddings`, `--listings`, `--splits`, and `--out-dir`.
- Produces: `<out-dir>/price_models.joblib` and `<out-dir>/price_metrics.json`.

- [x] **Step 1: Write the failing end-to-end artifact test**

```python
def test_run_training_writes_model_bundle_and_metrics(self):
    metrics = run_training(self.embeddings, self.listings, self.splits, self.output, n_estimators=20)
    self.assertTrue((self.output / "price_models.joblib").exists())
    self.assertEqual(json.loads((self.output / "price_metrics.json").read_text()), metrics)
    bundle = joblib.load(self.output / "price_models.joblib")
    self.assertEqual(bundle["quantiles"], [0.1, 0.5, 0.9])
```

- [x] **Step 2: Run the test to verify it fails**

Run: `python3 -m unittest tests.test_train_price.PriceModelTests.test_run_training_writes_model_bundle_and_metrics -v`
Expected: FAIL because `run_training` does not exist.

- [x] **Step 3: Implement artifact writing and CLI defaults**

```python
def run_training(embeddings_path, listings_path, splits_path, out_dir, n_estimators=200):
    data = load_training_data(embeddings_path, listings_path, splits_path)
    models = train_models(*data["train"], n_estimators=n_estimators)
    metrics = {name: evaluate(models, *data[name]) for name in ("val", "test")}
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"quantiles": list(QUANTILES), "models": models}, out_dir / "price_models.joblib")
    (out_dir / "price_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
```

Set CLI defaults to the three Task 2/3 paths in `data/processed` and `artifacts/price_model`.

- [x] **Step 4: Run all tests and CLI help**

Run: `python3 -m unittest discover -v`
Expected: all tests PASS.

Run: `python3 -m modeling.train_price --help`
Expected: exit code 0 and documented path options.
