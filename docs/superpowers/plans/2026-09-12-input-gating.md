# Input Gating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reject unusable or irrelevant uploads before appraisal and return the usable truck photos, confidence level, and machine-readable reasons.

**Architecture:** A new pipeline module first measures brightness and Laplacian sharpness with the existing Pillow and NumPy dependencies. Photos that pass quality checks are scored against truck and non-truck CLIP prompts using Task 3's frozen model; the combined gate accepts the request when at least one usable truck photo remains.

**Tech Stack:** Python 3, Pillow, NumPy, PyTorch, open_clip, pytest

**Spec:** `docs/main_specs.md`, Component 6

## Global Constraints

- Reuse `pipeline.extract_embeddings.MODEL_NAME` and `embed_images`; do not load a second vision model.
- Default quality thresholds are brightness 35 and Laplacian variance 40, chosen below the minimum or near-minimum values observed across 600 repository truck photos.
- Default semantic acceptance requires a truck-prompt similarity margin of at least 0.02; ten sampled truck photos scored 0.047–0.075.
- Reject a request only when no usable truck photo remains; otherwise preserve usable paths and report rejected photos as warnings.
- Confidence is low below three usable photos, medium when at least three survive but some inputs were rejected, and high when at least three inputs all pass.
- Duplicate-view detection remains excluded because the source plan marks it as stretch scope.

---

### Task 1: Deterministic image-quality checks

**Files:**
- Create: `pipeline/gate_images.py`
- Create: `tests/test_gate_images.py`

**Interfaces:**
- Produces: `assess_image_quality(path, dark_threshold=35.0, blur_threshold=40.0) -> dict` containing `path`, `brightness`, `sharpness`, and reason codes.

- [x] **Step 1: Write failing tests for dark, blurry, sharp, and unreadable inputs**

```python
def test_quality_checks_distinguish_dark_blurry_and_sharp_images(tmp_path):
    dark = save_image(tmp_path / "dark.png", np.zeros((32, 32), dtype=np.uint8))
    flat = save_image(tmp_path / "flat.png", np.full((32, 32), 180, dtype=np.uint8))
    sharp = save_image(tmp_path / "sharp.png", checkerboard())
    assert "too_dark" in assess_image_quality(dark)["reasons"]
    assert "too_blurry" in assess_image_quality(flat)["reasons"]
    assert assess_image_quality(sharp)["reasons"] == []

def test_quality_check_marks_unreadable_files(tmp_path):
    missing = tmp_path / "missing.png"
    assert assess_image_quality(missing)["reasons"] == ["unreadable"]
```

- [x] **Step 2: Run tests and verify RED**

Run: `.venv/bin/pytest tests/test_gate_images.py -q`
Expected: collection fails because `pipeline.gate_images` does not exist.

- [x] **Step 3: Implement the minimum quality metrics**

```python
def assess_image_quality(path, dark_threshold=DARK_THRESHOLD, blur_threshold=BLUR_THRESHOLD):
    try:
        with Image.open(path) as image:
            gray = np.asarray(image.convert("L"), dtype=np.float32)
    except Exception:
        return {"path": str(path), "brightness": None, "sharpness": None, "reasons": ["unreadable"]}
    center = gray[1:-1, 1:-1]
    laplacian = 4 * center - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    brightness, sharpness = float(gray.mean()), float(laplacian.var())
    reasons = []
    if brightness < dark_threshold:
        reasons.append("too_dark")
    if sharpness < blur_threshold:
        reasons.append("too_blurry")
    return {"path": str(path), "brightness": brightness, "sharpness": sharpness, "reasons": reasons}
```

- [x] **Step 4: Run tests and verify GREEN**

Run: `.venv/bin/pytest tests/test_gate_images.py -q`
Expected: quality tests pass.

### Task 2: CLIP relevance and combined decision

**Files:**
- Modify: `pipeline/gate_images.py`
- Modify: `tests/test_gate_images.py`

**Interfaces:**
- Produces: `similarity_margins(image_features, text_features, positive_count) -> numpy.ndarray`.
- Produces: `truck_similarity_margins(paths, model, preprocess, device) -> numpy.ndarray` using Task 3's encoder.
- Produces: `gate_images(paths, model, preprocess, device, ...) -> dict` containing `accepted`, `confidence`, `usable_paths`, `rejected`, `warnings`, and `reasons`.

- [x] **Step 1: Write failing tests for semantic margins and combined decisions**

```python
def test_similarity_margin_compares_best_positive_and_negative_prompt():
    images = np.array([[0.8, 0.1], [0.2, 0.7]])
    texts = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert similarity_margins(images, texts, positive_count=1).tolist() == pytest.approx([0.7, -0.5])

def test_gate_keeps_good_truck_photo_and_reports_rejections(monkeypatch, tmp_path):
    good, nontruck, dark = make_three_images(tmp_path)
    monkeypatch.setattr("pipeline.gate_images.truck_similarity_margins", lambda *args: np.array([0.05, -0.01]))
    result = gate_images([good, nontruck, dark], object(), object(), "cpu")
    assert result["accepted"] is True
    assert result["usable_paths"] == [good, nontruck]
    assert result["confidence"] == "low"
    assert {reason for item in result["rejected"] for reason in item["reasons"]} == {"too_dark", "too_blurry"}

def test_gate_rejects_when_no_usable_truck_photo_remains(monkeypatch, tmp_path):
    image = save_image(tmp_path / "sharp.png", checkerboard())
    monkeypatch.setattr("pipeline.gate_images.truck_similarity_margins", lambda *args: np.array([-0.01]))
    result = gate_images([image], object(), object(), "cpu")
    assert result["accepted"] is False
    assert result["reasons"] == ["no_usable_truck_photos"]
```

- [x] **Step 2: Run tests and verify RED**

Run: `.venv/bin/pytest tests/test_gate_images.py -q`
Expected: imports fail because semantic and combined-gate functions do not exist.

- [x] **Step 3: Implement CLIP prompt scoring and aggregation**

```python
def similarity_margins(image_features, text_features, positive_count):
    similarities = image_features @ text_features.T
    return similarities[:, :positive_count].max(axis=1) - similarities[:, positive_count:].max(axis=1)

@torch.no_grad()
def truck_similarity_margins(paths, model, preprocess, device):
    image_features = embed_images(paths, model, preprocess, device)
    tokens = open_clip.get_tokenizer(MODEL_NAME)(TRUCK_PROMPTS + NON_TRUCK_PROMPTS).to(device)
    text_features = model.encode_text(tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return similarity_margins(image_features, text_features.cpu().numpy(), len(TRUCK_PROMPTS))
```

Implement `gate_images` by quality-filtering first. Accept all quality-passing photos when at least one clear context view has a semantic margin of 0.02 or greater; otherwise reject the group as non-truck. This preserves close-up tire, interior, and cargo-box photos that are useful downstream but do not independently resemble a whole truck. Derive confidence from surviving and rejected counts.

- [x] **Step 4: Run Task 6 and complete regression tests**

Run: `.venv/bin/pytest tests/test_gate_images.py -q`
Expected: Task 6 tests pass.

Run: `.venv/bin/pytest -q`
Expected: Task 3 and Task 6 tests pass.
