# Kamion Truck Appraisal — 54Hackathon

An image-first commercial-truck appraisal prototype for the Kamion hackathon.
The goal is to estimate a defensible price range and condition notes from a
truck's listing photos while clearly communicating uncertainty.

## Repository layout

- `docs/` — hackathon brief and the proposed modeling plan.
- `data/commercial_truck_sales_100/` — completed-sale dataset used for
  development and evaluation.
  - `truck_sales_100.csv` and `.xlsx` contain one record for each sale.
  - `images/<item_id>/` contains six listing photos per truck.
  - `raw_json/` preserves the public source-listing data for each record.

## Dataset

The included dataset contains 100 completed Purple Wave sales: 50 box trucks
and 50 semi tractors. Sale prices include the buyer premium; taxes and
transport are excluded. See
[`data/commercial_truck_sales_100/README.txt`](data/commercial_truck_sales_100/README.txt)
for the source notes and field details.

## Getting started

From the repository root:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m pipeline.build_artifacts
./.venv/bin/python -m scripts.validate_pipeline
./.venv/bin/streamlit run app.py
./.venv/bin/pytest -q
```

The build writes cleaned listings, deterministic grouped splits, listing
embeddings, a training-only comparable index, condition tags and calibration
under `data/processed/`, plus the price models and held-out metrics under
`artifacts/price_model/`. The validator writes
`artifacts/validation_report.json`. These generated artifacts are ignored by
Git and should be rebuilt locally from the repository inputs.

## Reproduced build and validation

The default build on September 12, 2026 produced 98 usable listings with
78/10/10 train/validation/test splits. Listing and comparable embeddings were
finite and unit-normalized; their observed norm range was
0.99999991–1.00000013. The held-out price metrics were:

| Split | Pinball q10 | Pinball q50 | Pinball q90 | Median MAE | Median R² | Interval coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 601.5640516087482 | 1211.5036984275007 | 541.5432672699678 | 2423.0073968550014 | 0.5530310134687917 | 0.7 |
| Test | 1026.1801281635949 | 2726.5483061286986 | 1104.0211118473412 | 5453.096612257397 | -0.6581883714522285 | 0.4 |

The six-case validator used test listing `NJ9230` as its unseen US truck. Price
values below are the exact emitted USD low/median/high estimates; `null` means
the case was rejected before pricing.

| Case | Accepted | Confidence | USD low / median / high | Flagged conditions | Comparable IDs |
| --- | --- | --- | --- | --- | --- |
| Held-out US truck | Yes | high | 5981.004120305139 / 10033.626320079931 / 12730.348976604573 | none | FT6110, EF1319, YA2755 |
| Dark transform | No | low | null / null / null | none | none |
| Blurry transform | No | low | null / null / null | none | none |
| Motorcycle | No | low | null / null / null | none | none |
| Single truck photo | Yes | low | 0.0 / 13449.925816491394 / 31870.65940474376 | none | YA3412, FN2019, ER1987 |
| European truck | Yes | low | 1583.891187971476 / 14232.38779357344 / 17247.218844162668 | none | YA1038, ED5143, FC3251 |

Rejected cases reported `no_usable_truck_photos`. The accepted European image
only demonstrates that the truck-image gate recognizes it; the resulting USD
estimate is not calibrated for a European or Turkish market.

## Limitations

This is a research prototype trained on 100 completed US auction sales. It
outputs USD estimates and has no Turkish or European market calibration. The
small held-out samples and observed negative test R² mean the price estimates
should be treated as experimental, not as a professional valuation. Image
condition flags are model signals rather than an inspection or mechanical
diagnosis.
