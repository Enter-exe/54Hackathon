"""Adjust a model price interval using the image gate's confidence."""


CONFIDENCE_FACTORS = {"high": 1.0, "medium": 1.25, "low": 1.5}


def adjust_price_range(quantiles, gate_result):
    """Return a UI-ready price range, widened around q50 when confidence drops."""
    confidence = gate_result["confidence"]
    result = {
        "accepted": gate_result["accepted"],
        "price_low": None,
        "price_median": None,
        "price_high": None,
        "confidence": confidence,
        "warnings": gate_result["warnings"],
        "reasons": gate_result["reasons"],
    }
    if not result["accepted"]:
        return result

    lower, median, upper = map(float, quantiles)
    factor = CONFIDENCE_FACTORS[confidence]
    result.update(
        price_low=max(0.0, median - factor * (median - lower)),
        price_median=median,
        price_high=median + factor * (upper - median),
    )
    return result
