import socket

import pytest

from pipeline.listing_images import ListingExtractionError, extract_image_urls, validate_public_url


def test_extracts_purple_wave_listing_images_before_generic_assets():
    html = """
      <meta property="og:image" content="https://d323w7klwy72q3.cloudfront.net/i/a/FK3297.JPG">
      <img src="/logo.png">
      <img data-src="https://d323w7klwy72q3.cloudfront.net/i/a/FK3297A.JPG">
    """
    assert extract_image_urls(html, "https://www.purplewave.com/auction/item/FK3297")[:2] == [
        "https://d323w7klwy72q3.cloudfront.net/i/a/FK3297.JPG",
        "https://d323w7klwy72q3.cloudfront.net/i/a/FK3297A.JPG",
    ]


def test_extracts_ctt_and_generic_structured_images_without_price_fields():
    html = """
      <script type="application/ld+json">
        {"@type":"Product","offers":{"price":"72000"},
         "image":["https://cdn-media.tilabs.io/a.webp","/b.jpg"]}
      </script>
      <img srcset="/small.jpg 320w, /large.jpg 1280w">
    """
    urls = extract_image_urls(html, "https://www.commercialtrucktrader.com/listing/123")
    assert urls == [
        "https://cdn-media.tilabs.io/a.webp",
        "https://www.commercialtrucktrader.com/b.jpg",
        "https://www.commercialtrucktrader.com/small.jpg",
        "https://www.commercialtrucktrader.com/large.jpg",
    ]
    assert all("72000" not in url for url in urls)


def test_extract_image_urls_deduplicates_and_ignores_data_urls():
    html = """
      <meta property="og:image" content="/truck.jpg">
      <img src="/truck.jpg">
      <img src="data:image/png;base64,AAAA">
    """
    assert extract_image_urls(html, "https://example.com/listing") == [
        "https://example.com/truck.jpg"
    ]


def public_resolver(host, port, type=socket.SOCK_STREAM):
    return [(socket.AF_INET, type, 6, "", ("93.184.216.34", port))]


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "https://user:secret@example.com/truck",
        "https://example.com:8443/truck",
    ],
)
def test_validate_public_url_rejects_unsafe_shapes(url):
    with pytest.raises(ListingExtractionError) as error:
        validate_public_url(url, resolver=public_resolver)
    assert error.value.code == "unsafe_url"


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.1", "169.254.1.2", "::1", "fc00::1"])
def test_validate_public_url_rejects_non_global_addresses(address):
    def resolver(host, port, type=socket.SOCK_STREAM):
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        return [(family, type, 6, "", (address, port))]

    with pytest.raises(ListingExtractionError, match="public"):
        validate_public_url("https://example.com/truck", resolver=resolver)


def test_validate_public_url_returns_normalized_public_url_without_fragment():
    assert validate_public_url(
        "https://example.com/truck?id=4#photos", resolver=public_resolver
    ) == "https://example.com/truck?id=4"


class FakeResponse:
    def __init__(self, status_code=200, headers=None, chunks=()):
        self.status_code = status_code
        self.headers = headers or {"Content-Type": "text/html"}
        self._chunks = chunks
        self.encoding = "utf-8"

    def iter_content(self, chunk_size):
        yield from self._chunks

    def close(self):
        pass


def test_fetch_html_revalidates_redirect_targets():
    from pipeline.listing_images import fetch_html

    responses = iter([
        FakeResponse(302, {"Location": "http://127.0.0.1/admin"}),
    ])

    with pytest.raises(ListingExtractionError, match="public"):
        fetch_html(
            "https://example.com/truck",
            http_get=lambda *args, **kwargs: next(responses),
            resolver=lambda host, port, type=socket.SOCK_STREAM: [
                (socket.AF_INET, type, 6, "", (("127.0.0.1" if host == "127.0.0.1" else "93.184.216.34"), port))
            ],
        )


def test_fetch_html_rejects_oversized_body(monkeypatch):
    from pipeline.listing_images import fetch_html

    monkeypatch.setattr("pipeline.listing_images.MAX_HTML_BYTES", 8)
    response = FakeResponse(chunks=[b"123456", b"789"])

    with pytest.raises(ListingExtractionError, match="large"):
        fetch_html(
            "https://example.com/truck",
            http_get=lambda *args, **kwargs: response,
            resolver=public_resolver,
        )


def test_fetch_html_rejects_malformed_content_length():
    from pipeline.listing_images import fetch_html

    response = FakeResponse(headers={"Content-Type": "text/html", "Content-Length": "unknown"})

    with pytest.raises(ListingExtractionError, match="size"):
        fetch_html(
            "https://example.com/truck",
            http_get=lambda *args, **kwargs: response,
            resolver=public_resolver,
        )


def test_fetch_html_does_not_append_chunk_past_limit(monkeypatch):
    from pipeline.listing_images import fetch_html

    monkeypatch.setattr("pipeline.listing_images.MAX_HTML_BYTES", 8)
    response = FakeResponse(chunks=[b"123456789"])

    with pytest.raises(ListingExtractionError, match="large"):
        fetch_html(
            "https://example.com/truck",
            http_get=lambda *args, **kwargs: response,
            resolver=public_resolver,
        )


def test_fetch_html_returns_final_url_and_decoded_html():
    from pipeline.listing_images import fetch_html

    response = FakeResponse(chunks=[b"<html>truck</html>"])
    assert fetch_html(
        "https://example.com/truck",
        http_get=lambda *args, **kwargs: response,
        resolver=public_resolver,
    ) == ("https://example.com/truck", "<html>truck</html>")
