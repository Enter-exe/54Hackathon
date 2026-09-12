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
