import pytest

from modeling.confidence import adjust_price_range


@pytest.mark.parametrize(
    ("confidence", "expected_low", "expected_high"),
    [
        ("high", 24_000.0, 37_000.0),
        ("medium", 22_500.0, 38_750.0),
        ("low", 21_000.0, 40_500.0),
    ],
)
def test_adjust_price_range_scales_each_side_around_the_median(
    confidence, expected_low, expected_high
):
    gate_result = {
        "accepted": True,
        "confidence": confidence,
        "warnings": ["limited_photo_coverage"] if confidence == "low" else [],
        "reasons": [],
    }

    result = adjust_price_range([24_000, 30_000, 37_000], gate_result)

    assert result == {
        "accepted": True,
        "price_low": expected_low,
        "price_median": 30_000.0,
        "price_high": expected_high,
        "confidence": confidence,
        "warnings": gate_result["warnings"],
        "reasons": [],
    }


def test_adjust_price_range_suppresses_prices_for_rejected_uploads():
    gate_result = {
        "accepted": False,
        "confidence": "low",
        "warnings": ["some_photos_rejected"],
        "reasons": ["no_usable_truck_photos"],
    }

    result = adjust_price_range([24_000, 30_000, 37_000], gate_result)

    assert result == {
        "accepted": False,
        "price_low": None,
        "price_median": None,
        "price_high": None,
        "confidence": "low",
        "warnings": ["some_photos_rejected"],
        "reasons": ["no_usable_truck_photos"],
    }


def test_adjust_price_range_never_returns_a_negative_lower_price():
    gate_result = {
        "accepted": True,
        "confidence": "low",
        "warnings": ["limited_photo_coverage"],
        "reasons": [],
    }

    result = adjust_price_range([100, 1_000, 2_000], gate_result)

    assert result["price_low"] == 0.0
    assert result["price_median"] == 1_000.0
    assert result["price_high"] == 2_500.0
