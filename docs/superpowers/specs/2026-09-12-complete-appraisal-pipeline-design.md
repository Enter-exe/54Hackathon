# Complete Appraisal Pipeline Design

**Status:** Approved in chat on 2026-09-12

## Purpose

Finish the Kamion prototype as a real, locally runnable image-first truck
appraisal system. The result must use the checked-in sale data, produce actual
model artifacts, appraise new uploaded photos, explain the estimate with
condition flags and comparable sales, reject unusable inputs, and run through a
demo UI.

## Current State

Tasks 1–7 have implementation code, but the checked-in dataset does not match
the processing contract and no processed or trained artifacts exist. The
repository has 100 completed Purple Wave sales (50 box trucks and 50 truck
tractors), six images per sale, and verified sale-price labels. Tasks 8–10—
comparables retrieval, an end-to-end demo, and adversarial validation—are not
implemented.

## Chosen Approach

The Purple Wave dataset is the canonical training source. It is already local,
reproducible, and uses completed-sale prices, while the Commercial Truck Trader
scraper is externally fragile and provides asking prices. The existing scraper
will remain available, but completing the product will not depend on another
live scrape.

The prototype stays deliberately small: NumPy cosine search instead of FAISS,
one cached CLIP model, the existing LightGBM quantile models, and Streamlit for
the demo. Generated data and models remain local ignored artifacts and are
rebuilt with one documented command.

## Data and Artifact Build

A dataset preparation module will read
`data/commercial_truck_sales_100/truck_sales_100.csv`, normalize it to the
existing canonical columns, validate its referenced images, remove invalid
prices and 1st/99th-percentile outliers, and create a deterministic grouped
80/10/10 split. It will write:

- `data/processed/listings_clean.parquet`
- `data/processed/splits.json`

Canonical mappings include `item_id → ad_id`,
`sale_price_usd_including_buyer_premium → price`, `make → make_name`,
`model → model_name`, and `state → state_code`. Image paths will be stored as
repository-relative paths so the processed data can be rebuilt and used from
the repository root.

A single build command will then run the existing CLIP embedding/index stage,
condition calibration, and quantile price training. It will produce:

- `data/processed/embeddings.parquet`
- `data/processed/listing_embeddings.npz`
- `data/processed/comparables_index.npz`
- `data/processed/condition_tags.parquet`
- `data/processed/condition_calibration.json`
- `artifacts/price_model/price_models.joblib`
- `artifacts/price_model/price_metrics.json`

The build must fail clearly when a required source file, image, split, or
artifact is missing instead of silently producing an incomplete model.

## Appraisal Flow

One inference service will own artifact loading and expose an operation that
accepts uploaded image paths and returns a UI-ready dictionary.

1. Task 6 checks readability, darkness, blur, and truck relevance.
2. Rejected uploads return reasons and no price.
3. Usable images are embedded and mean-pooled with the frozen Task 3 encoder.
4. Task 4 predicts ordered q10/q50/q90 prices.
5. Task 7 widens the range according to Task 6 confidence.
6. Task 5 scores condition attributes with saved train-set thresholds.
7. Task 8 performs flat cosine search against the train-only comparables index
   and returns the top three sales with similarity, price, year, make, model,
   source URL, and a local thumbnail path.

The CLIP model and static artifacts are loaded once per application process.
The first implementation may encode a small upload set in more than one
existing stage; correctness and a working demo take priority over refactoring
the established Task 3/5/6 APIs. Shared-embedding optimization is deferred
unless measured latency makes the demo impractical.

## Demo UI

The Streamlit app will provide a multi-image uploader and one appraisal action.
It will show:

- a clear refusal with per-photo reasons when gating fails;
- low, median, and high prices in USD when accepted;
- High, Medium, or Low confidence plus warnings;
- condition flags with human-readable labels and scores;
- three comparable completed sales with thumbnail, vehicle details, sale
  price, similarity, and source link;
- an actionable setup message when generated artifacts are absent.

The interface is optimized for a four-minute live demo: one primary action,
plain status language, no optional typed vehicle metadata, no authentication,
and no database or remote API dependency during inference.

## Validation

Automated tests will cover dataset normalization/splitting, cosine comparable
ranking, artifact loading, accepted and rejected inference responses, and the
Streamlit empty/setup state. Existing Task 1–7 tests remain part of the full
suite.

The real pipeline will be built and exercised against:

- clear held-out truck photos;
- a darkened truck photo;
- a heavily blurred truck photo;
- a non-truck image;
- a single-photo upload to confirm Low confidence and interval widening;
- a licensed non-US-market truck photo when an attributable public fixture can
  be obtained.

The final report records actual validation/test metrics, including interval
coverage, without inventing a target result or hiding poor calibration. A
manual Streamlit smoke test must confirm that the app starts and renders its
initial state; at least one accepted appraisal and one rejected appraisal must
also be run through the inference service.

## Dependencies and Operation

A root `requirements.txt` will compose the existing pipeline/modeling
requirements and add Streamlit. The README will document:

1. environment installation;
2. the one-command artifact build;
3. the Streamlit launch command;
4. artifact locations and expected first-run CLIP download;
5. the dataset size and limitations.

The demo remains a US-dollar research prototype trained on 100 US auction
sales. It must state that it is not a production valuation and that European
or Turkish-market estimates may be poorly calibrated.

## Completion Criteria

The remaining work is complete only when all of the following are evidenced:

- the checked-in dataset is accepted by the processing pipeline;
- all expected processed and trained artifacts are generated from scratch;
- embeddings contain finite normalized vectors and splits remain disjoint;
- price metrics and condition calibration are written;
- a new upload can produce a gated, confidence-adjusted appraisal;
- top-three real comparable sales are returned;
- bad and irrelevant inputs suppress pricing with actionable reasons;
- the Streamlit app starts successfully and renders the required states;
- the complete automated suite passes;
- real/adversarial validation results and known limitations are documented.

## Explicit Non-Goals

- Reviving the CTT scraper as a prerequisite
- Fine-tuning CLIP
- FAISS or another vector database for 100 records
- Duplicate-angle detection beyond the existing limited-photo warning
- Currency conversion or Turkish-market calibration
- Authentication, deployment infrastructure, or a persistent backend
