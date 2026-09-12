# Complete Appraisal Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the checked-in Purple Wave sales and Tasks 1–7 into a reproducible, end-to-end truck appraisal demo with real artifacts, comparable sales, a Streamlit UI, and adversarial validation.

**Architecture:** Normalize the checked-in CSV into one canonical processed CSV, then use the existing CLIP, condition, gating, LightGBM, and confidence modules to build local NPZ/JSON/joblib artifacts. A small appraisal service composes those artifacts for a Streamlit UI, while NumPy performs exact cosine ranking over the 100-sale comparable index.

**Tech Stack:** Python 3.13, pandas, Pillow, NumPy, PyTorch, open_clip, LightGBM, scikit-learn, joblib, Streamlit, pytest

**Spec:** `docs/superpowers/specs/2026-09-12-complete-appraisal-pipeline-design.md`

## Global Constraints

- Use `data/commercial_truck_sales_100/` as the canonical dataset; do not require a live CTT scrape.
- Use CSV for tabular records, JSON for splits/calibration/metrics, NPZ for embeddings, and joblib for trained models.
- Remove Parquet and PyArrow from the required path.
- Preserve the grouped, deterministic 80/10/10 listing split with seed `0`.
- Reuse the existing frozen `ViT-B-32-quickgelu` CLIP model and existing Tasks 3–7; do not add another vision model.
- Use flat NumPy cosine ranking; do not add FAISS or a vector database for 100 records.
- Generated files under `data/processed/` and `artifacts/` stay ignored and must be reproducible with one command.
- The demo performs no network calls after the CLIP weights have been downloaded.
- Keep duplicate-angle detection, authentication, deployment infrastructure, currency conversion, and Turkish-market calibration out of scope.

---

### Task 1: Canonical CSV I/O and Parquet removal

**Files:**
- Create: `pipeline/data_io.py`
- Create: `tests/test_data_io.py`
- Modify: `pipeline/extract_embeddings.py`
- Modify: `pipeline/condition_assessment.py`
- Modify: `modeling/train_price.py`
- Modify: `pipeline/requirements.txt`
- Modify: `modeling/requirements.txt`
- Modify: `tests/test_extract_embeddings.py`
- Modify: `tests/test_condition_assessment.py`
- Modify: `tests/test_train_price.py`

**Interfaces:**
- Produces: `write_listings_csv(frame: pandas.DataFrame, path: Path) -> None`.
- Produces: `read_listings_csv(path: Path) -> pandas.DataFrame` with `ad_id` as strings and `image_paths` restored as `list[str]`.
- Changes the canonical listing path to `data/processed/listings_clean.csv` and condition output to `data/processed/condition_tags.csv`.
- Removes the unused `embeddings.parquet` output; `listing_embeddings.npz` remains the reusable embedding cache.

- [ ] **Step 1: Write the failing CSV round-trip test**

```python
def test_listing_csv_round_trip_preserves_ids_and_image_lists(tmp_path):
    frame = pd.DataFrame({
        "ad_id": ["FB5018"],
        "price": [10_780.0],
        "image_paths": [["data/images/a.jpg", "data/images/b.jpg"]],
    })
    path = tmp_path / "listings_clean.csv"

    write_listings_csv(frame, path)
    loaded = read_listings_csv(path)

    assert loaded.loc[0, "ad_id"] == "FB5018"
    assert loaded.loc[0, "image_paths"] == ["data/images/a.jpg", "data/images/b.jpg"]
```

- [ ] **Step 2: Run the test and verify RED**

Run: `./.venv/bin/pytest tests/test_data_io.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'pipeline.data_io'`.

- [ ] **Step 3: Implement JSON-backed list serialization inside CSV**

```python
import json
from pathlib import Path

import pandas as pd


def write_listings_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    output["image_paths"] = output["image_paths"].map(json.dumps)
    output.to_csv(path, index=False)


def read_listings_csv(path):
    frame = pd.read_csv(path, dtype={"ad_id": str})
    frame["image_paths"] = frame["image_paths"].map(json.loads)
    return frame
```

- [ ] **Step 4: Verify the CSV helper GREEN**

Run: `./.venv/bin/pytest tests/test_data_io.py -q`

Expected: `1 passed`.

- [ ] **Step 5: Change existing tests to require CSV artifacts**

Update the three integration fixtures to call `write_listings_csv`, replace
`listings_clean.parquet` with `listings_clean.csv`, read condition tags with
`pd.read_csv`, and remove the assertion for `embeddings.parquet`.

```python
self.listings_path = self.root / "listings_clean.csv"
write_listings_csv(listings, processed_dir / "listings_clean.csv")
condition_df = pd.read_csv(processed_dir / "condition_tags.csv")
```

- [ ] **Step 6: Run the migrated tests and verify RED**

Run: `./.venv/bin/pytest tests/test_extract_embeddings.py tests/test_condition_assessment.py tests/test_train_price.py -q`

Expected: failures reference the old `.parquet` paths or Parquet readers/writers.

- [ ] **Step 7: Migrate the three consumers and requirements**

Use `read_listings_csv` in `pipeline/extract_embeddings.py`,
`pipeline/condition_assessment.py`, and `modeling/train_price.py`. Remove the
`embeddings.parquet` write, write condition tags with `to_csv(index=False)`,
change CLI defaults to `.csv`, and delete `pyarrow` from both requirement files.

```python
df = read_listings_csv(processed_dir / "listings_clean.csv")
out_df.to_csv(processed_dir / "condition_tags.csv", index=False)
```

- [ ] **Step 8: Verify Task 1 GREEN and commit**

Run: `./.venv/bin/pytest -q`

Expected: all pre-existing tests plus the new CSV test pass.

```bash
git add pipeline/data_io.py pipeline/extract_embeddings.py pipeline/condition_assessment.py modeling/train_price.py pipeline/requirements.txt modeling/requirements.txt tests
git commit -m "refactor: use csv for processed listings"
```

---

### Task 2: Prepare the checked-in sale dataset

**Files:**
- Create: `pipeline/prepare_dataset.py`
- Create: `tests/test_prepare_dataset.py`

**Interfaces:**
- Consumes: `data/commercial_truck_sales_100/truck_sales_100.csv` and its pipe-separated, dataset-relative `image_paths` column.
- Produces: `prepare_sales_data(dataset_dir: Path, processed_dir: Path, repo_root: Path, seed: int = 0) -> tuple[pandas.DataFrame, dict[str, list[str]]]`.
- Writes: `listings_clean.csv` through `write_listings_csv` and `splits.json`.

- [ ] **Step 1: Write a failing normalization and split test**

Create 20 fixture sales with valid JPEGs, monotonically increasing positive
prices, and the real source column names. Assert that the extreme prices are
removed, mappings are canonical, paths resolve from `repo_root`, and split IDs
are disjoint and exhaustive.

```python
clean, splits = prepare_sales_data(dataset_dir, processed_dir, repo_root, seed=0)

assert {"ad_id", "price", "year", "make_name", "model_name", "image_paths"} <= set(clean)
assert clean["ad_id"].map(type).eq(str).all()
assert all((repo_root / path).is_file() for paths in clean["image_paths"] for path in paths)
assert not (set(splits["train"]) & set(splits["val"]))
assert set().union(*map(set, splits.values())) == set(clean["ad_id"])
assert read_listings_csv(processed_dir / "listings_clean.csv")["image_paths"].map(bool).all()
```

- [ ] **Step 2: Run the test and verify RED**

Run: `./.venv/bin/pytest tests/test_prepare_dataset.py -q`

Expected: collection fails because `pipeline.prepare_dataset` does not exist.

- [ ] **Step 3: Implement source normalization, validation, and splitting**

Map source fields explicitly, reject unreadable/missing images, deduplicate
`item_id`, keep positive finite sale prices, apply inclusive 1st/99th
percentile bounds, and use `random.Random(seed).shuffle(ids)`. Allocate counts
with `round(0.8 * n)` and `round(0.1 * n)`; the remainder is test.

```python
COLUMN_MAP = {
    "item_id": "ad_id",
    "sale_price_usd_including_buyer_premium": "price",
    "make": "make_name",
    "model": "model_name",
    "state": "state_code",
}

paths = [dataset_dir / value for value in row.image_paths.split("|")]
valid_paths = [path for path in paths if image_is_valid(path)]
relative_paths = [path.relative_to(repo_root).as_posix() for path in valid_paths]
```

The CLI defaults are:

```python
--dataset-dir data/commercial_truck_sales_100
--processed-dir data/processed
--repo-root .
--seed 0
```

- [ ] **Step 4: Verify Task 2 GREEN against fixtures and real source data**

Run: `./.venv/bin/pytest tests/test_prepare_dataset.py -q`

Expected: the fixture tests pass.

Run: `./.venv/bin/python -m pipeline.prepare_dataset`

Expected: `listings_clean.csv` and `splits.json` are written with nonempty,
disjoint train/val/test splits and every stored image path exists.

- [ ] **Step 5: Commit Task 2**

```bash
git add pipeline/prepare_dataset.py tests/test_prepare_dataset.py
git commit -m "feat: prepare completed-sale dataset"
```

---

### Task 3: One-command artifact build

**Files:**
- Create: `pipeline/build_artifacts.py`
- Create: `tests/test_build_artifacts.py`

**Interfaces:**
- Produces: `build_artifacts(repo_root: Path, n_estimators: int = 200) -> dict[str, Path]`.
- Calls, in order: `prepare_sales_data`, `pipeline.extract_embeddings.run`, `pipeline.condition_assessment.run`, and `modeling.train_price.run_training`.
- Verifies every required CSV/JSON/NPZ/joblib artifact exists before returning.

- [ ] **Step 1: Write a failing build-contract test**

Mock only the expensive CLIP/training stage boundaries. Each fake writes the
same complete file structure as its real counterpart; assert the returned map
contains every required artifact and every path exists.

```python
outputs = build_artifacts(tmp_path, n_estimators=5)

assert set(outputs) == {
    "listings", "splits", "embeddings", "comparables", "condition_tags",
    "condition_calibration", "price_models", "price_metrics",
}
assert all(path.exists() for path in outputs.values())
```

- [ ] **Step 2: Run the test and verify RED**

Run: `./.venv/bin/pytest tests/test_build_artifacts.py -q`

Expected: collection fails because `pipeline.build_artifacts` does not exist.

- [ ] **Step 3: Implement the build orchestrator and artifact check**

```python
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

missing = [name for name, path in outputs.items() if not path.exists()]
if missing:
    raise RuntimeError(f"artifact build incomplete: {missing}")
```

Run the module from `repo_root` so repository-relative image paths resolve.
Expose `--repo-root` and `--n-estimators` CLI arguments.

- [ ] **Step 4: Verify Task 3 GREEN and commit**

Run: `./.venv/bin/pytest tests/test_build_artifacts.py -q`

Expected: the build-contract test passes.

```bash
git add pipeline/build_artifacts.py tests/test_build_artifacts.py
git commit -m "feat: add reproducible artifact build"
```

---

### Task 4: Comparable-sale lookup

**Files:**
- Create: `modeling/comparables.py`
- Create: `tests/test_comparables.py`

**Interfaces:**
- Produces: `load_comparables(index_path: Path, listings_path: Path) -> tuple[dict[str, numpy.ndarray], dict[str, dict]]`.
- Produces: `find_comparables(query_embedding: numpy.ndarray, index: dict[str, numpy.ndarray], listings_by_id: dict[str, dict], k: int = 3) -> list[dict]`.
- Each result contains `ad_id`, `similarity`, `price`, `year`, `make_name`, `model_name`, `source_url`, and `thumbnail_path`.

- [ ] **Step 1: Write a failing exact-ranking test**

Use three hand-normalized two-dimensional vectors and a literal expected order.
The production mutation this catches is sorting ascending or ranking on a
non-normalized query.

```python
index = {
    "embeddings": np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]),
    "ad_id": np.array(["a", "b", "c"]),
}
rows = {
    key: {"ad_id": key, "price": price, "year": 2020,
          "make_name": "MAKE", "model_name": key,
          "source_url": f"https://example.com/{key}",
          "image_paths": [f"images/{key}.jpg"]}
    for key, price in zip("abc", [10_000, 20_000, 30_000])
}

results = find_comparables(np.array([0.9, 0.1]), index, rows, k=2)

assert [item["ad_id"] for item in results] == ["a", "b"]
assert results[0]["thumbnail_path"] == "images/a.jpg"
```

- [ ] **Step 2: Run the test and verify RED**

Run: `./.venv/bin/pytest tests/test_comparables.py -q`

Expected: collection fails because `modeling.comparables` does not exist.

- [ ] **Step 3: Implement loading and NumPy cosine ranking**

```python
def find_comparables(query_embedding, index, listings_by_id, k=3):
    query = np.asarray(query_embedding, dtype=np.float32)
    query /= np.linalg.norm(query) + 1e-8
    similarities = index["embeddings"] @ query
    order = np.argsort(similarities)[::-1][:k]
    results = []
    for position in order:
        ad_id = str(index["ad_id"][position])
        row = listings_by_id[ad_id]
        results.append({
            "ad_id": ad_id,
            "similarity": float(similarities[position]),
            "price": float(row["price"]),
            "year": int(row["year"]),
            "make_name": row["make_name"],
            "model_name": row["model_name"],
            "source_url": row["source_url"],
            "thumbnail_path": row["image_paths"][0],
        })
    return results
```

`load_comparables` loads NPZ with `allow_pickle=True`, loads canonical listings
with `read_listings_csv`, converts IDs to strings, and rejects mismatched index
row counts.

- [ ] **Step 4: Add boundary assertions and verify GREEN**

Add tests for `k` larger than the index and a zero query vector raising
`ValueError("query embedding must be nonzero")`.

Run: `./.venv/bin/pytest tests/test_comparables.py -q`

Expected: all comparable tests pass.

- [ ] **Step 5: Commit Task 4**

```bash
git add modeling/comparables.py tests/test_comparables.py
git commit -m "feat: return nearest comparable sales"
```

---

### Task 5: End-to-end appraisal service

**Files:**
- Create: `pipeline/appraise.py`
- Create: `tests/test_appraise.py`

**Interfaces:**
- Produces: `artifacts_ready(repo_root: Path) -> bool`.
- Produces: `AppraisalEngine.from_artifacts(repo_root: Path) -> AppraisalEngine`.
- Produces: `AppraisalEngine.appraise(image_paths: list[str]) -> dict`.
- Accepted results contain the Task 7 price fields plus `condition` and `comparables`; rejected results contain `None` prices, reasons, warnings, an empty condition map, and an empty comparable list.

- [ ] **Step 1: Write failing accepted/rejected service tests**

Patch only the existing expensive model operations (`gate_images`,
`pooled_listing_embedding`, `predict_quantiles`, and `assess_images`) while
using the real Task 7 adjustment and Task 8 ranking. Assert complete
consumer-visible results rather than mock call counts.

```python
accepted = engine.appraise(["truck-a.jpg", "truck-b.jpg", "truck-c.jpg"])
assert accepted["accepted"] is True
assert accepted["price_median"] == 30_000.0
assert accepted["confidence"] == "high"
assert len(accepted["comparables"]) == 3
assert set(accepted["condition"]) == {"rust", "body_damage", "tire_wear", "interior_wear"}

rejected = engine.appraise(["motorcycle.jpg"])
assert rejected["accepted"] is False
assert rejected["price_median"] is None
assert rejected["comparables"] == []
assert rejected["condition"] == {}
```

- [ ] **Step 2: Run tests and verify RED**

Run: `./.venv/bin/pytest tests/test_appraise.py -q`

Expected: collection fails because `pipeline.appraise` does not exist.

- [ ] **Step 3: Implement artifact loading and appraisal composition**

`from_artifacts` loads the frozen model/preprocessor, price bundle, condition
thresholds/text embeddings, Task 5 photo gate, and comparable index/listings.
The instance retains these resources so Streamlit can cache one engine.

```python
gate_result = gate_images(image_paths, self.model, self.preprocess, self.device)
if not gate_result["accepted"]:
    return {
        **adjust_price_range([], gate_result),
        "condition": {},
        "comparables": [],
    }

embedding = pooled_listing_embedding(
    gate_result["usable_paths"], self.model, self.preprocess, self.device,
    gate=self.photo_gate,
)
prediction = predict_quantiles(self.price_models, embedding[None, :])[0]
condition = assess_images(
    gate_result["usable_paths"], self.model, self.preprocess, self.device,
    self.condition_text_pairs, thresholds=self.condition_thresholds,
    gate=self.photo_gate,
)
return {
    **adjust_price_range(prediction, gate_result),
    "condition": condition or {},
    "comparables": find_comparables(
        embedding, self.comparable_index, self.listings_by_id, k=3
    ),
}
```

If Task 5 filtering leaves no pooled embedding, return the same rejected shape
with reason `no_usable_truck_photos`.

- [ ] **Step 4: Implement and test artifact readiness**

```python
def artifacts_ready(repo_root):
    root = Path(repo_root)
    return all((root / relative).is_file() for relative in REQUIRED_ARTIFACTS)
```

Assert false for an empty temporary root and true after creating every listed
file.

- [ ] **Step 5: Verify Task 5 GREEN and commit**

Run: `./.venv/bin/pytest tests/test_appraise.py -q`

Expected: appraisal-service tests pass.

Run: `./.venv/bin/pytest -q`

Expected: the complete suite passes.

```bash
git add pipeline/appraise.py tests/test_appraise.py
git commit -m "feat: compose end-to-end truck appraisals"
```

---

### Task 6: Streamlit demo

**Files:**
- Create: `requirements.txt`
- Create: `app.py`
- Create: `tests/test_app.py`

**Interfaces:**
- Launches with: `./.venv/bin/streamlit run app.py`.
- Uses `@st.cache_resource` to load one `AppraisalEngine`.
- Accepts multiple JPG/JPEG/PNG/WEBP files and renders the complete appraisal result.

- [ ] **Step 1: Add the root environment contract and install Streamlit**

```text
-r pipeline/requirements.txt
-r modeling/requirements.txt
streamlit
```

Run: `./.venv/bin/pip install -r requirements.txt`

Expected: Streamlit installs without changing the existing declared model dependencies.

- [ ] **Step 2: Write a failing initial-state app test**

```python
from streamlit.testing.v1 import AppTest


def test_app_renders_upload_flow_without_loading_models():
    app = AppTest.from_file("app.py").run(timeout=10)
    assert not app.exception
    assert app.title[0].value == "Kamion Truck Appraisal"
    assert app.file_uploader[0].label == "Truck photos"
    assert app.button[0].label == "Appraise truck"
```

- [ ] **Step 3: Run the app test and verify RED**

Run: `./.venv/bin/pytest tests/test_app.py -q`

Expected: failure because `app.py` does not exist.

- [ ] **Step 4: Implement the single-screen demo**

Use `st.set_page_config`, a title and concise prototype disclaimer, a multiple
file uploader, and an appraisal button. Do not load the engine until the button
is used. Save uploads in `tempfile.TemporaryDirectory`, preserving safe numeric
filenames and extensions.

Render rejected results with `st.error` plus per-photo reason labels. Render
accepted results with three price metrics, a confidence badge, warnings,
condition rows, and three comparable cards containing `st.image`, vehicle
details, sale price, similarity, and `st.link_button`.

```python
@st.cache_resource
def get_engine():
    return AppraisalEngine.from_artifacts(Path.cwd())

if appraise_clicked:
    if not artifacts_ready(Path.cwd()):
        st.error("Model artifacts are missing. Run: python -m pipeline.build_artifacts")
    elif not uploads:
        st.warning("Upload at least one truck photo.")
    else:
        result = get_engine().appraise(saved_paths)
        render_result(result)
```

- [ ] **Step 5: Verify Task 6 GREEN and commit**

Run: `./.venv/bin/pytest tests/test_app.py -q`

Expected: the Streamlit initial-state test passes without loading CLIP.

Run: `./.venv/bin/streamlit run app.py --server.headless true --server.port 8501`

Expected: the server reaches its healthy startup state; terminate it after an
HTTP request to `http://localhost:8501/_stcore/health` returns `ok`.

```bash
git add requirements.txt app.py tests/test_app.py
git commit -m "feat: add truck appraisal demo"
```

---

### Task 7: Adversarial validation, real build, and documentation

**Files:**
- Create: `scripts/validate_pipeline.py`
- Create: `tests/test_validation.py`
- Create: `tests/fixtures/ATTRIBUTION.md`
- Add: `tests/fixtures/motorcycle.jpg`
- Add: `tests/fixtures/european_truck.jpg`
- Modify: `README.md`

**Interfaces:**
- Produces: `make_dark(source: Path, destination: Path) -> None`.
- Produces: `make_blurry(source: Path, destination: Path) -> None`.
- Produces: `validate_pipeline(repo_root: Path, engine: AppraisalEngine | None = None) -> dict` and writes `artifacts/validation_report.json`.
- Uses the first test-split listing as the unseen US truck case and the two attributed public fixtures for non-truck and non-US cases.

- [ ] **Step 1: Acquire and attribute offline fixtures**

Download these two 960-pixel Wikimedia Commons derivatives and record the
source-page URL, author, license, derivative URL, and retrieval date in
`tests/fixtures/ATTRIBUTION.md`:

- `motorcycle.jpg`: “A motorcycle.jpg,” Summering2018, CC BY-SA 4.0,
  source `https://commons.wikimedia.org/wiki/File:A_motorcycle.jpg`, download
  `https://commons.wikimedia.org/wiki/Special:Redirect/file/A%20motorcycle.jpg?width=960`.
- `european_truck.jpg`: “Mercedes-Benz Actros truck.jpg,” High Contrast,
  CC BY 3.0 DE, source
  `https://commons.wikimedia.org/wiki/File:Mercedes-Benz_Actros_truck.jpg`,
  download
  `https://commons.wikimedia.org/wiki/Special:Redirect/file/Mercedes-Benz%20Actros%20truck.jpg?width=960`.

Verify both downloaded files with Pillow before using or staging them.

- [ ] **Step 2: Write failing transformation and report-shape tests**

```python
make_dark(source, dark_path)
make_blurry(source, blurry_path)

assert assess_image_quality(dark_path)["brightness"] < DARK_THRESHOLD
assert assess_image_quality(blurry_path)["sharpness"] < BLUR_THRESHOLD

report = validate_pipeline(tmp_repo, engine=fake_engine)
assert set(report) == {
    "held_out_truck", "dark", "blurry", "motorcycle",
    "single_photo", "non_us_truck",
}
```

The fake engine returns complete real appraisal-shaped dictionaries; assertions
target the report aggregation and disk write, not fake call counts.

- [ ] **Step 3: Run tests and verify RED**

Run: `./.venv/bin/pytest tests/test_validation.py -q`

Expected: collection fails because `scripts.validate_pipeline` does not exist.

- [ ] **Step 4: Implement deterministic validation cases**

Use Pillow `ImageEnhance.Brightness(...).enhance(0.03)` for darkness and
`ImageFilter.GaussianBlur(radius=25)` for blur. Run the six cases through one
loaded `AppraisalEngine`; record acceptance, confidence, reasons, price fields,
condition flags, and comparable IDs as JSON-serializable values.

```python
assert report["held_out_truck"]["accepted"]
assert not report["dark"]["accepted"]
assert not report["blurry"]["accepted"]
assert not report["motorcycle"]["accepted"]
assert report["single_photo"]["confidence"] == "low"
assert report["non_us_truck"]["accepted"] in (True, False)
```

If the non-US photograph is rejected, preserve that observed result in the
report and document it as a known limitation rather than weakening the truck
gate solely to pass the fixture.

- [ ] **Step 5: Document exact setup, build, run, and limitations**

Update `README.md` with these commands and artifact contracts:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m pipeline.build_artifacts
./.venv/bin/python -m scripts.validate_pipeline
./.venv/bin/streamlit run app.py
./.venv/bin/pytest -q
```

State that the model is a research prototype trained on 100 US auction sales,
outputs USD estimates, and is not a professional valuation. Include the actual
validation and price-metric values after running the real build.

- [ ] **Step 6: Run the real artifact build and verification**

Run: `./.venv/bin/python -m pipeline.build_artifacts`

Verify with a read-only check that every expected artifact exists, the listing
and embedding row counts agree, all embeddings are finite and unit-normalized,
and split IDs are disjoint.

Run: `./.venv/bin/python -m scripts.validate_pipeline`

Expected: the JSON report is written; held-out/single/non-US outcomes and all
adversarial rejection outcomes are visible and documented exactly as observed.

- [ ] **Step 7: Run final automated and UI verification**

Run: `./.venv/bin/pytest -q`

Run: `./.venv/bin/python -m compileall -q pipeline modeling scripts tests app.py`

Run: `./.venv/bin/pip check`

Run the Streamlit health smoke test from Task 6 once more against the built
artifacts. Expected: tests pass, compilation exits zero, dependencies are
consistent, and the server health endpoint returns `ok`.

- [ ] **Step 8: Audit scope and commit Task 7**

Confirm the final diff contains only the approved pipeline, inference, demo,
validation, fixtures, tests, requirements, and documentation changes. Do not
force-add ignored generated artifacts.

```bash
git add README.md scripts/validate_pipeline.py tests/test_validation.py tests/fixtures
git commit -m "test: validate complete appraisal workflow"
```
