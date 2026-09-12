# Kamion Challenge — Revised 24h Plan

## What changed from v1 and why

v1 (price-only) satisfies "a price estimate, with a range" but misses two things the brief explicitly judges: **condition assessment** and **knowing its limits**. Both are added here using the *same* CLIP embeddings already being computed for pricing — no second model, no extra training, minimal added time.

---

## Architecture (v2)

```
uploaded photos
      │
      ├─► quality/gating check (blur, brightness, truck-vs-not) ──► if fails: "can't assess, need better photos"
      │
      ▼
  frozen CLIP ViT-B/32 encoder ──► per-image embedding
      │
      ├─► mean-pool ──► LightGBM quantile models (q10/q50/q90) ──► price range
      │
      ├─► cosine similarity vs. condition text prompts ──► condition tags (rust, dents, tire wear, interior)
      │
      └─► cosine similarity vs. training-set embedding index ──► top-3 nearest comparable listings (price + thumbnail)
      │
      ▼
  confidence adjustment (fewer/lower-quality photos → wider range, lower confidence label)
      │
      ▼
  demo UI: price range + confidence + condition flags + comparables + (if gated) "need better photos" message
```

---

## Components

### 1. Scraper — Commercial Truck Trader, box trucks (~3h)
- Fields: price, listing ID, full-size image URLs, year/make/model/box length (kept for debugging/comparables display, not model input)
- `requests` + BeautifulSoup first; Playwright only if listings are JS-rendered
- Target a few thousand listings — don't chase volume past that, it's not the bottleneck
- Cap concurrency to avoid getting blocked mid-hackathon
- Output: Parquet/JSON metadata + images on disk keyed by listing ID

### 2. Cleaning + split (~1h)
- Drop $0 / "call for price" listings
- Drop price outliers outside ~1st–99th percentile
- Drop dead/placeholder images (hash-check against known "no photo" stock images)
- Dedupe by listing ID
- Single grouped 80/10/10 split by listing ID (no k-fold needed — encoder is frozen, no leakage path)

### 3. Embedding extraction + comparables index (~1.5h)
- Run frozen CLIP ViT-B/32 (via `open_clip` or `transformers`) over every training image once
- Mean-pool per listing, cache to disk (Parquet/npz) keyed by listing ID — must be cached, will be reused across iterations
- Fallback: ResNet50 (torchvision, ImageNet weights) if CLIP setup is fussy — don't debug environment issues for long
- Build a similarity index over all training-listing pooled embeddings (a flat numpy array + cosine similarity, or FAISS if available) — this is the same embeddings, just also kept for nearest-neighbor lookup at inference time

### 4. Price model — LightGBM quantile regression (~1.5h)
- Three models: q=0.1, 0.5, 0.9, trained on cached embeddings → price
- Sanity-check calibration: does the q10–q90 interval actually contain ~80% of val-set true prices?
- Report pinball loss per quantile and median MAE/R² on test split — don't chase a target number

### 5. Condition assessment — zero-shot CLIP, no training (~1.5h)
- Fixed set of short text prompts, each describing one condition attribute, e.g.:
  - "a photo of a rusty truck body"
  - "a photo of a truck with dented or damaged panels"
  - "a photo of worn or damaged tires"
  - "a photo of a clean, well-maintained truck interior"
  - "a photo of a truck cab interior showing wear or damage"
  - "a photo of a clean, undamaged truck exterior"
- Encode all prompts once with CLIP's text encoder
- At inference, compute cosine similarity between each uploaded image's embedding and each prompt embedding
- Threshold or rank to produce condition tags (e.g., "rust detected," "tires appear worn," "no major body damage visible")
- No labeled condition data needed — this is the key trick that avoids the manual labeling effort cut from the original spec

### 6. Gating / limits-awareness (~1.5h)
- **Truck-vs-not check:** cosine similarity between uploaded image and prompts like "a photo of a truck" vs. "a photo of a motorcycle," "a photo of a person," "a photo of an unclear or irrelevant scene." If truck similarity doesn't clearly dominate, refuse to price and say why.
- **Basic image quality check (OpenCV, free):**
  - Blur: Laplacian variance below a threshold → flag as too blurry
  - Darkness: mean pixel brightness below a threshold → flag as too dark
- **Combined logic:** if any check fails, respond with what's wrong ("this doesn't look like a truck," "photo too dark/blurry, please retake") instead of forcing a price out
- **View coverage (stretch, only if time allows):** if only one photo is provided, or all photos are near-duplicate angles (very high mutual cosine similarity), flag "limited view — estimate less reliable" rather than silently pricing off partial information

### 7. Confidence scaling (~30min, folded into #4/#6)
- Base rule: fewer images and/or lower quality/coverage → multiply the q10–q90 interval width by a scaling factor, and attach a confidence label (High / Medium / Low)
- Doesn't need to be sophisticated — a simple heuristic (e.g., <3 images or any quality flag triggers a 1.5x interval widening) is enough to demonstrate the system knows when it's less sure, which is explicitly judged

### 8. Comparables ("reasoning you can check") (~1h)
- At inference, embed the uploaded photos, mean-pool, and do a cosine similarity lookup against the training-set index built in step 3
- Surface the top 2-3 most similar real listings: thumbnail + actual price + (year/make/model if available)
- This gives a checkable, concrete answer to "why this price" without an LLM narrating a number it didn't actually compute — avoids looking like a wrapper

### 9. Demo UI (~3-4h)
- Streamlit (fastest to build): photo upload → runs full pipeline → displays:
  - Price range + median + confidence label
  - Condition tags
  - Comparable listings with thumbnails
  - If gated: clear message about what's wrong with the input, no price shown
- Prioritize this being demoable end-to-end over polish

### 10. Adversarial testing / buffer (remaining time — do not skip)
- Explicitly test against: a non-truck photo, a blurry/dark photo, a single-photo-only upload, and (if possible) a non-US-market truck photo, to see how the system behaves and to have an honest answer ready if judges' live photos hit these cases
- This step is what the judging criteria "does it know its limits" and "does it survive contact with photos it's never seen" actually test — treat it as core work, not optional polish

---

## Explicitly cut / deprioritized (same reasoning as v1, still applies)

- Fine-tuning the encoder — frozen CLIP only
- Attention-pooling multi-image aggregation — mean pooling only
- 5-fold CV / leakage-safe Stage1→Stage2 split — not needed, encoder is frozen and never trained on this data
- Manual condition-score labeling — replaced entirely by zero-shot CLIP prompts, no labels needed
- eBay calibration / second data source — single source (Commercial Truck Trader) only
- Optional typed-detail input (year/make/km) — dropped; brief wants photos to do the work, and UI time is better spent on gating/condition logic
- LLM-generated reasoning paragraph — replaced by comparables, which are checkable and don't risk looking like a thin wrapper

---

## Known risk: market mismatch

Training data is US-market (Commercial Truck Trader, USD). Kamion is a Turkish freight platform — live judging photos may show different brands, currencies, or truck configurations common in Turkey/Europe but rare in the US training set. This isn't fixable by re-scoping data collection in 24h. Mitigations:
- Explicitly test the system against a few non-US truck photos before the demo to know how it fails
- Be upfront in the pitch about the training distribution, and treat the gating logic gracefully flagging "outside known distribution" as a feature demonstration, not a failure
- Frame comparables honestly — if the nearest matches are a poor fit, that itself is useful signal to surface rather than hide

---

## Revised time budget (~24h)

| Step | Time |
|---|---|
| Scraper | 3h |
| Cleaning + split | 1h |
| Embedding extraction + comparables index | 1.5h |
| LightGBM quantile training | 1.5h |
| Condition assessment (zero-shot CLIP) | 1.5h |
| Gating + quality checks | 1.5h |
| Confidence scaling | 0.5h |
| Comparables lookup | 1h |
| Demo UI | 3-4h |
| Adversarial testing / buffer | remaining (~8-9h) |

---

## Verification checklist

- [ ] Scraped rows spot-checked (price + images look correct) before scaling up
- [ ] Embeddings verified non-NaN after extraction
- [ ] Calibration check: q10–q90 interval covers ~80% of held-out true prices
- [ ] Zero-shot condition tags spot-checked against a few known-damaged and known-clean training images
- [ ] Gating correctly rejects a non-truck photo and a heavily blurred/dark photo
- [ ] Confidence label visibly changes between a full clean photo set and a single low-quality photo
- [ ] End-to-end: upload a real, unseen box truck photo and confirm a sane, non-degenerate output
- [ ] At least one non-US-market truck photo tested to know failure behavior ahead of live judging