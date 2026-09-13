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

Use `truck_sales_100.csv` as the tabular index, matching each row's `item_id`
to its folder in `images/`. Keep all images from a listing together when
creating train, validation, and test splits to prevent leakage.

## Running the demo

Install the deployed UI/model dependencies and start Streamlit from the
repository root:

```bash
python -m pip install -r ui/requirements.txt
streamlit run ui/app.py
```

For a fresh setup without generated model artifacts, install the separate
pipeline and training dependencies (including `pyarrow`). The pipeline requires
`data/raw/listings.parquet` with listing IDs (`ad_id`), prices, and vehicle fields,
plus matching photos at `data/images/<ad_id>/*.webp`. The
[`scraper/scrape_ctt.py`](scraper/scrape_ctt.py) acquisition script produces both;
[`scraper/jsonl_to_parquet.py`](scraper/jsonl_to_parquet.py) can convert an existing
`data/raw/scraped_raw.jsonl` listing dump but does not download its images.
The completed-sale CSV under `data/commercial_truck_sales_100/` uses a different
layout and must be converted before these pipeline commands can consume it.
Once those inputs are present, build the artifacts before starting Streamlit:

```bash
python -m pip install -r pipeline/requirements.txt -r modeling/requirements.txt
python pipeline/clean_split.py
python -m pipeline.extract_embeddings
python -m pipeline.condition_assessment
python -m modeling.train_price
```

Browser rendering for inline JavaScript listing galleries is optional. Enable it by
installing the scraper dependencies and Chromium:

```bash
python -m pip install -r scraper/requirements.txt
python -m playwright install chromium
```

The app accepts either uploaded truck photos or a public listing URL. URL
appraisal extracts photos only; asking prices and current bids are not model
inputs. Purple Wave and Commercial Truck Trader receive dedicated parsing,
with generic structured-image extraction for other public listing pages.
The browser evaluates bounded HTML offline and blocks all external scripts,
images, media, and WebSocket requests. Pages requiring external JavaScript or
network calls may need manual photo upload. Blocked, unsafe, oversized, or
excessively redirected pages also return to manual upload without a browser retry.
