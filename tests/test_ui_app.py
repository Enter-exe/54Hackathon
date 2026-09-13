from contextlib import nullcontext
from pathlib import Path

import pytest

from pipeline.listing_images import ListingExtractionError
from ui import app
from ui.app import appraise_listing_url, listing_error_message


class FakeStreamlit:
    def __init__(
        self,
        *,
        source_mode="Listing link",
        submitted=False,
        listing_url="",
        uploaded_files=None,
    ):
        self.source_mode = source_mode
        self.submitted = submitted
        self.listing_url = listing_url
        self.uploaded_files = uploaded_files
        self.calls = []

    def _record(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))

    def title(self, *args, **kwargs):
        self._record("title", *args, **kwargs)

    def caption(self, *args, **kwargs):
        self._record("caption", *args, **kwargs)

    def error(self, *args, **kwargs):
        self._record("error", *args, **kwargs)

    def info(self, *args, **kwargs):
        self._record("info", *args, **kwargs)

    def success(self, *args, **kwargs):
        self._record("success", *args, **kwargs)

    def warning(self, *args, **kwargs):
        self._record("warning", *args, **kwargs)

    def image(self, *args, **kwargs):
        self._record("image", *args, **kwargs)

    def radio(self, *args, **kwargs):
        self._record("radio", *args, **kwargs)
        return self.source_mode

    def form(self, *args, **kwargs):
        self._record("form", *args, **kwargs)
        return nullcontext()

    def text_input(self, *args, **kwargs):
        self._record("text_input", *args, **kwargs)
        return self.listing_url

    def form_submit_button(self, *args, **kwargs):
        self._record("form_submit_button", *args, **kwargs)
        return self.submitted

    def spinner(self, *args, **kwargs):
        self._record("spinner", *args, **kwargs)
        return nullcontext()

    def file_uploader(self, *args, **kwargs):
        self._record("file_uploader", *args, **kwargs)
        return self.uploaded_files


class UploadedFile:
    name = "truck.jpg"

    def getbuffer(self):
        return memoryview(b"truck-photo")


def call_names(fake):
    return [name for name, _, _ in fake.calls]


def accepted_result():
    return {
        "gate": {"accepted": True},
        "price": {"price_median": 12_000},
        "predicted_class": [],
        "predicted_make": [],
        "condition": {},
        "comparables": [],
    }


def patch_result_renderers(monkeypatch):
    rendered = []
    monkeypatch.setattr(app, "render_price", lambda result: rendered.append("price"))
    monkeypatch.setattr(
        app,
        "render_predicted_specs",
        lambda predicted_class, predicted_make: rendered.append("specs"),
    )
    monkeypatch.setattr(
        app, "render_condition", lambda result: rendered.append("condition")
    )
    monkeypatch.setattr(
        app, "render_comparables", lambda result: rendered.append("comparables")
    )
    return rendered


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


def test_listing_error_message_uses_fixed_copy_for_unknown_domain_error():
    error = ListingExtractionError("future_error", "**untrusted listing message**")

    assert listing_error_message(error) == (
        "We couldn't extract photos from that listing automatically."
    )


def test_main_link_mode_previews_source_and_uses_shared_result_renderers(
    monkeypatch,
):
    fake = FakeStreamlit(
        source_mode="Listing link",
        submitted=True,
        listing_url="https://example.com/truck",
    )
    resources = object()
    observed = {}

    def fake_appraise(url, output_dir, actual_resources):
        photo = Path(output_dir) / "listing.jpg"
        photo.write_bytes(b"listing-photo")
        observed.update(
            url=url,
            output_dir=Path(output_dir),
            resources=actual_resources,
            photo=photo,
        )
        return (
            {
                "source_url": url,
                "source_host": "example.com",
                "image_paths": [str(photo)],
                "warnings": ["One photo was skipped."],
            },
            accepted_result(),
        )

    monkeypatch.setattr(app, "st", fake)
    monkeypatch.setattr(app, "cached_load_resources", lambda: resources)
    monkeypatch.setattr(app, "appraise_listing_url", fake_appraise)
    rendered = patch_result_renderers(monkeypatch)

    app.main()

    assert observed["url"] == "https://example.com/truck"
    assert observed["resources"] is resources
    assert not observed["output_dir"].exists()
    assert "file_uploader" not in call_names(fake)
    assert (
        "radio",
        (
            "How would you like to provide the truck photos?",
            ["Listing link", "Upload photos"],
        ),
        {"horizontal": True},
    ) in fake.calls
    assert (
        "success",
        ("Extracted 1 photos from example.com.",),
        {},
    ) in fake.calls
    image_call = next(call for call in fake.calls if call[0] == "image")
    assert image_call[1][0] == [str(observed["photo"])]
    assert image_call[2]["caption"] == ["Listing photo 1"]
    assert rendered == ["price", "specs", "condition", "comparables"]


def test_main_upload_mode_skips_link_extraction_and_uses_shared_result_renderers(
    monkeypatch,
):
    uploaded = UploadedFile()
    fake = FakeStreamlit(source_mode="Upload photos", uploaded_files=[uploaded])
    resources = object()
    observed = {}

    def fake_run_appraisal(paths, actual_resources):
        observed["paths"] = [Path(path) for path in paths]
        observed["contents"] = [path.read_bytes() for path in observed["paths"]]
        observed["resources"] = actual_resources
        return accepted_result()

    monkeypatch.setattr(app, "st", fake)
    monkeypatch.setattr(app, "cached_load_resources", lambda: resources)
    monkeypatch.setattr(
        app,
        "appraise_listing_url",
        lambda *args, **kwargs: pytest.fail("link extraction must not run"),
    )
    monkeypatch.setattr(app, "run_appraisal", fake_run_appraisal)
    rendered = patch_result_renderers(monkeypatch)

    app.main()

    assert "form" not in call_names(fake)
    assert "file_uploader" in call_names(fake)
    assert observed["contents"] == [b"truck-photo"]
    assert observed["resources"] is resources
    assert not observed["paths"][0].exists()
    assert next(call for call in fake.calls if call[0] == "image")[1][0] == [uploaded]
    assert rendered == ["price", "specs", "condition", "comparables"]


def test_main_listing_error_uses_fixed_copy_and_offers_manual_fallback(monkeypatch):
    fake = FakeStreamlit(
        source_mode="Listing link",
        submitted=True,
        listing_url="https://example.com/truck",
    )

    def fail_extraction(*args, **kwargs):
        raise ListingExtractionError("future_error", "**untrusted listing message**")

    monkeypatch.setattr(app, "st", fake)
    monkeypatch.setattr(app, "cached_load_resources", object)
    monkeypatch.setattr(app, "appraise_listing_url", fail_extraction)
    monkeypatch.setattr(
        app, "render_price", lambda result: pytest.fail("results must not render")
    )

    app.main()

    assert (
        "error",
        ("We couldn't extract photos from that listing automatically.",),
        {},
    ) in fake.calls
    assert (
        "info",
        ("Switch to Upload photos to continue manually.",),
        {},
    ) in fake.calls
    assert "**untrusted listing message**" not in repr(fake.calls)


def test_main_rejected_result_uses_existing_rejection_renderer(monkeypatch):
    fake = FakeStreamlit(
        source_mode="Listing link",
        submitted=True,
        listing_url="https://example.com/truck",
    )
    rejected = {"accepted": False, "rejected": []}
    observed = []

    def fake_appraise(url, output_dir, resources):
        photo = Path(output_dir) / "listing.jpg"
        photo.write_bytes(b"listing-photo")
        return (
            {
                "source_url": url,
                "source_host": "example.com",
                "image_paths": [str(photo)],
                "warnings": [],
            },
            {"gate": rejected},
        )

    monkeypatch.setattr(app, "st", fake)
    monkeypatch.setattr(app, "cached_load_resources", object)
    monkeypatch.setattr(app, "appraise_listing_url", fake_appraise)
    monkeypatch.setattr(app, "render_rejection", observed.append)
    monkeypatch.setattr(
        app, "render_price", lambda result: pytest.fail("price must not render")
    )

    app.main()

    assert observed == [rejected]
