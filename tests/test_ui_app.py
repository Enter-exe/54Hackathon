import pytest

from pipeline.listing_images import ListingExtractionError
from ui.app import appraise_listing_url, listing_error_message


def test_appraise_listing_url_passes_only_extracted_paths_to_model(tmp_path):
    observed = {}

    def extractor(url, output_dir, **kwargs):
        return {
            "source_url": url,
            "source_host": "example.com",
            "image_paths": ["/tmp/a.jpg", "/tmp/b.jpg"],
            "warnings": [],
        }

    def appraiser(paths, resources):
        observed["paths"] = paths
        observed["resources"] = resources
        return {"price": {"price_median": 12_000}}

    resources = {"model": object()}
    extraction, appraisal = appraise_listing_url(
        "https://example.com/truck",
        tmp_path,
        resources,
        extractor=extractor,
        appraiser=appraiser,
        browser_renderer=None,
    )

    assert observed["paths"] == ["/tmp/a.jpg", "/tmp/b.jpg"]
    assert observed["resources"] is resources
    assert set(extraction) == {"source_url", "source_host", "image_paths", "warnings"}
    assert appraisal["price"]["price_median"] == 12_000


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("unsafe_url", "Enter a public HTTP or HTTPS listing URL."),
        ("unavailable", "We couldn't reach that listing. Check the URL and try again."),
        ("blocked", "That listing site blocked automatic photo extraction."),
        ("redirect", "That listing redirected too many times."),
        ("unsupported", "That URL did not return a supported listing page."),
        ("too_large", "That listing is too large to process automatically."),
        ("no_images", "We couldn't find usable truck photos on that listing."),
        ("browser_unavailable", "Automatic browser extraction is unavailable."),
        ("browser_failed", "We couldn't render that listing automatically."),
    ],
)
def test_listing_error_message_maps_failures_to_actionable_copy(code, expected):
    error = ListingExtractionError(code, "low-level message")

    assert listing_error_message(error) == expected


def test_listing_error_message_preserves_unknown_domain_error():
    error = ListingExtractionError("future_error", "Try a different listing.")

    assert listing_error_message(error) == "Try a different listing."
