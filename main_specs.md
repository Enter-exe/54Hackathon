# Box Truck Price Prediction from Images — Full Project Plan

## 0. Goal Statement

Predict the price of a used box truck **from images alone** (no tabular specs at inference time). Output should be a **price distribution**, not a point estimate, since images cannot see mileage, mechanical condition, or title status.

Architecture: two-stage pipeline.
- **Stage 1** — supervised visual model that learns to recognize box-truck attributes from photos (make/model/year, box length, body type, condition).
- **Stage 2** — a price model trained on Stage 1's outputs (embeddings + predicted attributes) that outputs a price distribution.

Specs are used **only as training supervision in Stage 1** — never as model input at inference.

---

## 1. Data Sources

### 1.1 Primary training data (on-market / asking price)
**Source:** Commercial Truck Trader — dedicated Box Truck category, dealer + private listings, large volume (tens of thousands of active box truck listings at any time).

Fields to scrape per listing:
- Price (asking)
- Year, make, model
- Box length (ft), box type (dry van / reefer / flatbed / other)
- GVWR class
- Mileage, condition (if stated)
- All image URLs (not just thumbnail)
- Listing ID, dealer vs private seller, region, date posted

**Secondary on-market source (optional, for volume):** TruckPaper — similar commercial vehicle coverage, dealer-heavy.

### 1.2 Calibration / correction data (sold price)
**Source:** eBay Motors — Commercial Vans & Box Trucks category, filtered to completed + sold listings (`LH_Complete=1&LH_Sold=1`).

This gives real transaction prices + photos, without the enthusiast-inflation bias seen in collector-vehicle auction sites (e.g. Bring a Trailer). Expect a much smaller sample than Commercial Truck Trader (low hundreds to low thousands).

Fields to scrape:
- Final sold price
- Sold date
- Same vehicle attributes as above where available
- All image URLs
- Listing type (auction vs Best Offer vs fixed price)

### 1.3 Explicitly rejected / deprioritized sources
- **Craigslist Cars/Trucks (Kaggle)** — box trucks are not well represented in this dataset (dominated by pickups/passenger vehicles). Not used as primary source.
- **Copart / IAAI (salvage auctions)** — real sold prices but heavily biased toward wrecked/repossessed vehicles. Not representative of general box truck market. Not used unless building a separate "damaged vehicle" sub-model later.
- **Bring a Trailer** — sold prices are enthusiast/collector-inflated, wrong population for ordinary box trucks. Not used.

---

## 2. Data Cleaning

Applied to both CTT and eBay data, before anything touches the model:

1. **Dedupe listings** — same VIN or same image hashes reposted at different times/prices. Keep only the most recent, or treat carefully if you want repost price-drop signal (skip for v1).
2. **Drop dead image URLs** — verify each URL resolves and returns a valid image before including.
3. **Price sanity filtering** — drop $0, "call for price" placeholders, and statistical outliers (e.g., outside the 1st–99th percentile) unless manually verified as real.
4. **Category verification** — confirm listing is actually a box truck (dry van / cutaway / cube van body), not a pickup, flatbed-only chassis, or unrelated commercial vehicle mislabeled by the source site.
5. **Image quality filter** — drop listings with only 1 low-res stock/placeholder image if you can detect it (hash against known "no photo available" placeholder images used by these sites).
6. **Region/time normalization** — tag each listing with scrape date and region; used later to check for distribution shift, not necessarily to correct prices directly in v1.

---

## 3. Label Taxonomy (Stage 1 supervision targets)

These become the auxiliary head targets. All are derivable from the scraped tabular fields — you already have "ground truth" for training, even though the model input is images only.

| Head | Type | Classes / Range |
|---|---|---|
| Make | Classification | Isuzu, Freightliner, Ford (Transit/E-Series/F-59), International, Hino, Chevrolet/GMC (LCF), Mack MD, etc. |
| Model / chassis family | Classification | Model-specific, grouped if long-tail |
| Model year (or year-bucket) | Classification (bucketed, e.g. 3-year bins) or ordinal regression | 2005–present |
| Box length | Regression or bucketed classification | 10–26 ft typical range |
| Box type | Classification | Dry van, reefer, flatbed w/ box, cutaway, other |
| GVWR class | Classification | Class 3–7 |
| Visible condition score | Ordinal regression | 1 (heavy damage/rust) – 5 (like new) |
| Damage flags (optional, multi-label) | Multi-label binary | Dents, rust, roof damage, door damage, decal/wrap present |

Condition score labels: since this isn't in the raw scrape, generate weak labels via a simple heuristic + manual spot-check (e.g., sample 500 listings, manually rate 1-5, then either hand-label a larger batch or bootstrap using price residual after controlling for year/mileage as a rough proxy — treat this as noisy supervision, refine iteratively).

---

## 4. Data Splitting (leakage-safe)

**Rule: split by listing ID, never by image.** All images belonging to one truck must land in the same split.

- Stage 1: single split — 70% train / 15% val / 15% test, grouped by listing ID. Full k-fold not used here (too expensive relative to benefit given large dataset size).
- Stage 2: 5-fold grouped cross-validation (grouped by listing ID) — cheap to run since Stage 2 model is lightweight (GBM/MLP on embeddings, not raw images).

**Critical leakage fix — Stage 1 → Stage 2 handoff:**
Do NOT generate Stage 2 training embeddings from a Stage 1 model that was trained on those same listings. Approach:
1. Train Stage 1 once on ~80% of listings.
2. Generate embeddings for the held-out ~20% using that trained model.
3. Train and validate Stage 2 (including its 5-fold CV) *only* on that held-out 20% (out-of-fold embeddings).
4. If more Stage 2 data is needed, repeat with 2–3 non-overlapping splits of Stage 1 training data instead of a full k-fold retrain of Stage 1.

---

## 5. Stage 1: Visual Spec Predictor

### Architecture
- **Backbone:** ConvNeXt-Base or ViT-B/16, ImageNet-pretrained, fine-tuned.
- **Multi-image aggregation:** encode each of a listing's images independently through the shared backbone, then aggregate with attention pooling (learned weights per image) rather than simple averaging, so damage visible in only one angle isn't diluted.
- **Heads (on top of pooled embedding):**
  - Cross-entropy heads: make, model, year-bucket, box type, GVWR class
  - Regression/ordinal heads: box length, condition score
  - Optional multi-label head: damage flags (binary cross-entropy per flag)

### Loss
Weighted sum of all head losses. Start with equal weighting, then tune weights based on which heads are lagging in validation performance (common to slightly upweight condition score since it's the noisiest label and most price-relevant).

### Training config (starting point)
- Optimizer: AdamW, cosine LR schedule with warmup
- Batch size: as large as GPU memory allows (multi-image aggregation is memory-heavy — consider gradient checkpointing)
- Augmentation: random crop, color jitter (careful — condition-relevant cues like rust color must survive), horizontal flip (fine for exterior shots)
- Early stopping on validation macro-F1 (classification heads) + MAE (regression heads), combined into one tracked metric

### Output
For each listing: a pooled embedding vector + the predicted spec labels (used as-is, not ground truth, since at inference there is no ground truth).

---

## 6. Stage 2: Price Distribution Predictor

### Inputs
- Stage 1 pooled embedding vector (out-of-fold, per Section 4)
- Stage 1 **predicted** labels (make, model, year-bucket, box length, box type, condition score) — predicted, not true, since inference won't have true labels

### Model choice
**Gradient-boosted trees (LightGBM/XGBoost) with quantile regression**, not an LLM and not a plain neural regression head:
- GBMs handle mixed embedding + categorical-prediction inputs well without much feature engineering
- Quantile regression is natively supported and simple to calibrate

### Output format
Predict multiple quantiles (e.g., 10th, 25th, 50th, 75th, 90th percentile price) via separate quantile-loss models, or a single model with multiple quantile objectives (LightGBM supports this natively). This gives you a usable price *range*, e.g., "$18,200–$24,600, median $20,900" instead of a false-precision point estimate.

### Cross-validation
5-fold grouped CV (Section 4) to get a reliable estimate of pinball loss and to check calibration stability across folds.

---

## 7. Sold-Data Calibration (eBay)

Given eBay sold data is much smaller than the CTT training set, do **not** naively mix it into Stage 2 training with a hand-tuned sample weight. Instead:

**Two-phase approach (preferred):**
1. Train Stage 2 fully on CTT (ask-price) data as above.
2. Freeze Stage 1 backbone. Lightly fine-tune only the Stage 2 price head on eBay sold data (low learning rate, few epochs / few boosting rounds) — this recalibrates the price mapping toward true transaction prices without letting a small, potentially skewed sample distort the visual feature learning.

**Calibration-only alternative (lower risk, simpler):**
Don't fine-tune anything. Instead, use eBay sold data purely as a validation/correction set:
- Check whether CTT-trained model's predicted quantile intervals actually contain the true eBay sold prices at the expected rate (e.g., does the 80% interval catch 80% of real outcomes?)
- If there's a systematic bias (e.g., model predicts consistently X% above actual sold price), apply a simple post-hoc linear correction (e.g., `corrected_price = a * predicted_price + b`, fit via regression against the eBay sold set) rather than retraining anything

**Before either approach:** compare eBay-sold vs CTT truck populations on visible attributes (age distribution, box type mix, box length distribution) to check for distribution shift. If eBay skews meaningfully different (e.g., older/cheaper trucks), account for that before trusting the calibration blindly.

---

## 8. Evaluation

### Stage 1 (per head)
- Classification heads: accuracy, macro-F1 (watch for class imbalance — e.g., Isuzu/Freightliner likely dominate make distribution)
- Regression/ordinal heads: MAE, and for condition score specifically, check correlation with price residuals as a sanity check

### Stage 2
- **Pinball (quantile) loss** per quantile level
- **Calibration plot** — for each nominal interval (e.g., 80%), check empirical coverage on held-out data
- **R² / MAE on median prediction** — headline number, but interpret against the realistic ceiling below
- **Segmented evaluation** — break out performance by box type and year-bucket to see where the model is weakest (e.g., likely worse on unusual configurations with fewer training examples)

### Realistic ceiling
Expect explained variance (R²) in roughly the 0.5–0.65 range for image-only prediction, even with a well-built pipeline — mileage, mechanical condition, and title status are invisible to images and account for a meaningful chunk of remaining price variance. Treat this range as "working as intended," not as a sign of a broken model.

---

## 9. Tooling / Stack

- **Scraping:** Python (requests/Playwright for JS-heavy pages), or Apify actors for CTT/eBay if scraping infra is a bottleneck
- **Data storage:** Parquet for tabular metadata, cloud/local blob storage for images, keyed by listing ID
- **Stage 1 training:** PyTorch + timm (backbone) or torchvision
- **Stage 2 training:** LightGBM (native quantile objective support)
- **Experiment tracking:** Weights & Biases or MLflow — important given multiple heads and two-stage pipeline complexity
- **Compute:** Stage 1 needs GPU (multi-image aggregation is memory-heavy); Stage 2 is CPU-feasible

---

## 10. Suggested Milestones

1. **Data pipeline** — scrape + clean CTT and eBay data, verify image URLs, build listing-grouped splits
2. **Label taxonomy finalized** — especially condition score labeling strategy (manual sample + heuristic bootstrap)
3. **Stage 1 v1** — train on make/model/year/box-type/box-length only (skip condition initially), validate visual recognition works before adding noisier heads
4. **Stage 1 v2** — add condition score + damage flags once base recognition is solid
5. **Stage 2 v1** — quantile regression on Stage 1 outputs, CTT-only, establish baseline pinball loss
6. **Calibration** — bring in eBay sold data via chosen method (two-phase fine-tune or post-hoc correction), re-evaluate
7. **Segmented error analysis** — identify weak spots (box types, years, regions) and decide if targeted data collection is worth it

---

## Key Risks to Watch

- **Leakage** between Stage 1 and Stage 2 embeddings (Section 4) — the most likely silent bug in this pipeline
- **Condition score label quality** — this is your noisiest supervision signal and also your most price-relevant one; worth disproportionate manual QA time
- **Distribution shift** between CTT (training) and eBay (calibration) populations — check before trusting calibration correction
- **Class imbalance** in make/model — a few manufacturers likely dominate box truck listings; long-tail models may need grouping or dropping