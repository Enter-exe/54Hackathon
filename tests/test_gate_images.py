import numpy as np
import pytest
from PIL import Image

from pipeline.gate_images import assess_image_quality, gate_images, similarity_margins


def save_image(path, pixels):
    Image.fromarray(pixels.astype(np.uint8)).save(path)
    return str(path)


def checkerboard(size=32):
    return (np.indices((size, size)).sum(axis=0) % 2) * 255


def test_quality_checks_distinguish_dark_blurry_and_sharp_images(tmp_path):
    dark = save_image(tmp_path / "dark.png", np.zeros((32, 32)))
    flat = save_image(tmp_path / "flat.png", np.full((32, 32), 180))
    sharp = save_image(tmp_path / "sharp.png", checkerboard())

    assert "too_dark" in assess_image_quality(dark)["reasons"]
    assert "too_blurry" in assess_image_quality(flat)["reasons"]
    assert assess_image_quality(sharp)["reasons"] == []


def test_quality_check_marks_unreadable_files(tmp_path):
    missing = tmp_path / "missing.png"

    assert assess_image_quality(missing)["reasons"] == ["unreadable"]


def test_similarity_margin_compares_best_positive_and_negative_prompt():
    images = np.array([[0.8, 0.1], [0.2, 0.7]])
    texts = np.array([[1.0, 0.0], [0.0, 1.0]])

    margins = similarity_margins(images, texts, positive_count=1)

    assert margins.tolist() == [pytest.approx(0.7), pytest.approx(-0.5)]


def test_gate_keeps_detail_photos_when_one_context_view_is_a_truck(monkeypatch, tmp_path):
    good = save_image(tmp_path / "good.png", checkerboard())
    nontruck = save_image(tmp_path / "nontruck.png", 255 - checkerboard())
    dark = save_image(tmp_path / "dark.png", np.zeros((32, 32)))
    monkeypatch.setattr(
        "pipeline.gate_images.truck_similarity_margins",
        lambda *args: np.array([0.05, -0.01]),
    )

    result = gate_images([good, nontruck, dark], object(), object(), "cpu")

    assert result["accepted"] is True
    assert result["usable_paths"] == [good, nontruck]
    assert result["confidence"] == "low"
    rejected_reasons = {
        reason for item in result["rejected"] for reason in item["reasons"]
    }
    assert rejected_reasons == {"too_dark", "too_blurry"}


def test_gate_rejects_when_no_usable_truck_photo_remains(monkeypatch, tmp_path):
    image = save_image(tmp_path / "sharp.png", checkerboard())
    monkeypatch.setattr(
        "pipeline.gate_images.truck_similarity_margins",
        lambda *args: np.array([-0.01]),
    )

    result = gate_images([image], object(), object(), "cpu")

    assert result["accepted"] is False
    assert result["reasons"] == ["no_usable_truck_photos"]


def test_gate_confidence_reflects_coverage_and_rejections(monkeypatch, tmp_path):
    images = [
        save_image(tmp_path / f"sharp-{index}.png", checkerboard())
        for index in range(3)
    ]
    dark = save_image(tmp_path / "dark.png", np.zeros((32, 32)))
    monkeypatch.setattr(
        "pipeline.gate_images.truck_similarity_margins",
        lambda *args: np.full(3, 0.05),
    )

    high = gate_images(images, object(), object(), "cpu")
    medium = gate_images([*images, dark], object(), object(), "cpu")

    assert high["confidence"] == "high"
    assert medium["confidence"] == "medium"
