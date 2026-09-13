import builtins
from io import BytesIO
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest
import requests
from PIL import Image

from pipeline.listing_images import (
    ListingExtractionError,
    download_images,
    extract_image_urls,
    extract_listing_images,
    fetch_html,
    render_html_with_playwright,
    validate_public_url,
)


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


def test_extract_image_urls_normalizes_scheme_and_fragment_before_deduplication():
    html = """
      <meta property="og:image" content="HTTPS://example.com/truck.jpg#one">
      <img src="https://example.com/truck.jpg#two">
      <img src="DATA:image/png;base64,AAAA">
    """
    assert extract_image_urls(html, "https://example.com/listing") == [
        "https://example.com/truck.jpg"
    ]


def test_extracts_json_ld_with_media_type_parameters():
    html = """
      <script type="application/ld+json; charset=utf-8">
        {"image":"/structured.jpg"}
      </script>
    """
    assert extract_image_urls(html, "https://example.com/listing") == [
        "https://example.com/structured.jpg"
    ]


def test_malformed_json_ld_does_not_hide_valid_open_graph_image():
    html = """
      <script type="application/ld+json">{not-json</script>
      <meta property="og:image" content="/truck.jpg">
    """
    assert extract_image_urls(html, "https://example.com/listing") == [
        "https://example.com/truck.jpg"
    ]


def test_site_preferences_require_boundaries_and_parsed_image_hosts():
    html = """
      <img src="/generic.jpg">
      <img src="https://d323w7klwy72q3.cloudfront.net/i/a/listing.jpg">
      <img src="https://example.com/cloudfront.net/i/a/fake.jpg">
    """
    assert extract_image_urls(html, "https://notpurplewave.com/listing")[:2] == [
        "https://notpurplewave.com/generic.jpg",
        "https://d323w7klwy72q3.cloudfront.net/i/a/listing.jpg",
    ]

    html = """
      <img src="/generic.jpg">
      <img src="https://cdn-media.tilabs.io/listing.jpg">
    """
    assert extract_image_urls(html, "https://notcommercialtrucktrader.com/listing") == [
        "https://notcommercialtrucktrader.com/generic.jpg",
        "https://cdn-media.tilabs.io/listing.jpg",
    ]


def public_resolver(host, port, type=socket.SOCK_STREAM):
    return [(socket.AF_INET, type, 6, "", ("93.184.216.34", port))]


class FakeBrowserResponse:
    def __init__(self, status=200, headers=None):
        self.status = status
        self.headers = headers or {}


class FakeRoute:
    def __init__(self, url, status=200, headers=None, redirected_from=None, automatic_redirects=()):
        self.request = SimpleNamespace(url=url, redirected_from=redirected_from)
        self.response = FakeBrowserResponse(status, headers)
        self.aborted = False
        self.fulfilled = False
        self.fetch_max_redirects = None
        self.fetch_timeout = None
        self.automatic_redirects = automatic_redirects
        self.contacted = []

    def fetch(self, *, max_redirects, timeout):
        self.contacted.append(self.request.url)
        self.fetch_max_redirects = max_redirects
        self.fetch_timeout = timeout
        return self.response

    def abort(self):
        self.aborted = True

    def fulfill(self, *, response):
        assert response is self.response
        self.fulfilled = True
        # Chromium follows a fulfilled redirect without invoking the route again.
        self.contacted.extend(self.automatic_redirects)


class FakeWebSocket:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakePage:
    def __init__(
        self, *, routes=None, popup_routes=None, web_sockets=None, content_route=None
    ):
        self.url = "https://example.com/final"
        self.routes = routes or []
        self.popup_routes = popup_routes or []
        self.web_sockets = web_sockets or []
        self.content_route = content_route
        self.context = None
        self.closed = False
        self.goto_error = None
        self.rendered_html = '<meta property="og:image" content="/truck.jpg">'
        self.received_html = None

    def set_content(self, html, wait_until, timeout):
        self.received_html = html
        for route in [*self.routes, *self.popup_routes]:
            self.context.route_handler(route)
        for web_socket in self.web_sockets:
            self.context.web_socket_handler(web_socket)
        if self.goto_error:
            raise self.goto_error

    def goto(self, url, wait_until, timeout):
        self.requested_url = url
        routes = self.routes or [FakeRoute(url)]
        for route in [*routes, *self.popup_routes]:
            self.context.route_handler(route)
            if route.aborted:
                raise RuntimeError("request aborted")
        for web_socket in self.web_sockets:
            self.context.web_socket_handler(web_socket)
        if self.goto_error:
            raise self.goto_error

    def wait_for_load_state(self, state, timeout):
        pass

    def content(self):
        if self.content_route is not None:
            self.context.route_handler(self.content_route)
        return self.rendered_html

    def close(self):
        self.closed = True


class FakeContext:
    def __init__(self, page, fail_new_page=False):
        self.page = page
        self.fail_new_page = fail_new_page
        self.route_handler = None
        self.web_socket_handler = None
        self.closed = False

    def route(self, pattern, handler):
        self.route_handler = handler

    def route_web_socket(self, pattern, handler):
        self.web_socket_handler = handler

    def new_page(self):
        assert self.route_handler is not None
        if self.fail_new_page:
            raise RuntimeError("page creation failed")
        self.page.context = self
        return self.page

    def close(self):
        self.closed = True


class FakeBrowser:
    def __init__(self, page, fail_new_page=False, fail_context=False):
        self.context = FakeContext(page, fail_new_page)
        self.fail_context = fail_context
        self.context_options = None
        self.closed = False

    def new_context(self, user_agent, accept_downloads, service_workers, offline=False):
        if self.fail_context:
            raise RuntimeError("context creation failed")
        self.context_options = {
            "user_agent": user_agent,
            "accept_downloads": accept_downloads,
            "service_workers": service_workers,
            "offline": offline,
        }
        return self.context

    def close(self):
        self.closed = True


class FakePlaywrightManager:
    def __init__(self, page, fail_new_page=False, fail_context=False):
        self.browser = FakeBrowser(page, fail_new_page, fail_context)
        self.playwright = SimpleNamespace(
            chromium=SimpleNamespace(launch=lambda headless: self.browser)
        )

    def __enter__(self):
        return self.playwright

    def __exit__(self, exc_type, exc, traceback):
        pass


def fake_playwright_factory(page):
    return lambda: FakePlaywrightManager(page)


def public_resolver_with_localhost(host, port, type=socket.SOCK_STREAM):
    address = "127.0.0.1" if host == "127.0.0.1" else "93.184.216.34"
    return [(socket.AF_INET, type, 6, "", (address, port))]


@pytest.fixture
def browser_http(monkeypatch):
    def get(url, **kwargs):
        assert kwargs["allow_redirects"] is False
        assert kwargs["stream"] is True
        return FakeResponse(chunks=[b"<html><body>Inline content</body></html>"])

    monkeypatch.setattr("pipeline.listing_images.requests.get", get)
    return get


def test_playwright_renderer_never_hands_browser_a_redirect(browser_http):
    route = FakeRoute(
        "https://a.example/script.js", status=302,
        headers={"location": "https://b.example/script.js"},
        automatic_redirects=["https://b.example/script.js", "http://127.0.0.1/private"],
    )
    page = FakePage(routes=[route])
    render_html_with_playwright(
        "https://example.com/start", resolver=public_resolver_with_localhost,
        playwright_factory=fake_playwright_factory(page),
    )
    assert route.contacted == []
    assert route.aborted
    assert not route.fulfilled


def test_playwright_renderer_bounds_main_html_before_browser(monkeypatch):
    monkeypatch.setattr(
        "pipeline.listing_images.requests.get",
        lambda *args, **kwargs: FakeResponse(chunks=[b"x" * (5 * 1024 * 1024 + 1)]),
    )
    with pytest.raises(ListingExtractionError) as error:
        render_html_with_playwright(
            "https://example.com/start", resolver=public_resolver,
            playwright_factory=lambda: pytest.fail("oversized HTML reached browser"),
        )
    assert error.value.code == "too_large"


def test_playwright_renderer_rejects_two_public_hops_then_private_before_contact(monkeypatch):
    responses = iter([
        FakeResponse(302, {"Location": "https://b.example/truck"}),
        FakeResponse(302, {"Location": "http://127.0.0.1/private"}),
    ])
    requested = []

    def get(url, **kwargs):
        requested.append(url)
        assert kwargs["allow_redirects"] is False
        return next(responses)

    monkeypatch.setattr("pipeline.listing_images.requests.get", get)
    with pytest.raises(ListingExtractionError) as error:
        render_html_with_playwright(
            "https://a.example/truck", resolver=public_resolver_with_localhost,
            playwright_factory=lambda: pytest.fail("unsafe redirect reached browser"),
        )
    assert error.value.code == "unsafe_url"
    assert requested == ["https://a.example/truck", "https://b.example/truck"]


def test_playwright_renderer_returns_fetched_url_and_offline_html(browser_http, monkeypatch):
    page = FakePage()
    page.url = "about:blank"
    manager = FakePlaywrightManager(page)
    responses = iter([
        FakeResponse(302, {"Location": "/final"}),
        FakeResponse(chunks=[b"<html>Inline content</html>"]),
    ])
    monkeypatch.setattr("pipeline.listing_images.requests.get", lambda *a, **k: next(responses))
    final_url, html = render_html_with_playwright(
        "https://example.com/start",
        resolver=public_resolver,
        playwright_factory=lambda: manager,
    )
    assert final_url == "https://example.com/final"
    assert "truck.jpg" in html
    assert "Inline content" in page.received_html
    assert manager.browser.context_options["accept_downloads"] is False
    assert manager.browser.context_options["service_workers"] == "block"
    assert manager.browser.context_options["offline"] is True
    assert page.closed is True
    assert manager.browser.context.closed is True
    assert manager.browser.closed is True


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "http://127.0.0.1/private", "https://example.com/script.js",
    "https://images.example.com/truck.jpg", "https://example.com/large-video.mp4",
])
def test_playwright_renderer_aborts_all_resources_before_fetch(url, browser_http):
    route = FakeRoute(url)
    page = FakePage(routes=[route])
    render_html_with_playwright(
        "https://example.com/start",
        resolver=public_resolver_with_localhost,
        playwright_factory=fake_playwright_factory(page),
    )
    assert route.aborted
    assert route.contacted == []
    assert not route.fulfilled


def test_playwright_renderer_blocks_late_and_popup_requests(browser_http):
    late = FakeRoute("http://127.0.0.1/private")
    popup = FakeRoute("https://popup.example.com/start")
    page = FakePage(content_route=late, popup_routes=[popup])
    render_html_with_playwright(
        "https://example.com/start", resolver=public_resolver_with_localhost,
        playwright_factory=fake_playwright_factory(page),
    )
    assert late.aborted and popup.aborted
    assert late.contacted == popup.contacted == []


def test_playwright_renderer_blocks_web_sockets(browser_http):
    web_socket = FakeWebSocket()
    page = FakePage(web_sockets=[web_socket])
    render_html_with_playwright(
        "https://example.com/start", resolver=public_resolver,
        playwright_factory=fake_playwright_factory(page),
    )
    assert web_socket.closed is True


def test_playwright_renderer_bounds_redirect_chains(monkeypatch):
    requested = []

    def get(url, **kwargs):
        requested.append(url)
        assert kwargs["allow_redirects"] is False
        return FakeResponse(302, {"Location": f"/redirect-{len(requested)}"})

    monkeypatch.setattr("pipeline.listing_images.requests.get", get)
    with pytest.raises(ListingExtractionError) as error:
        render_html_with_playwright(
            "https://example.com/start", resolver=public_resolver,
            playwright_factory=lambda: pytest.fail("redirect limit reached browser"),
        )
    assert error.value.code == "redirect"
    assert len(requested) == 4


def test_playwright_renderer_bounds_rendered_html(browser_http):
    page = FakePage()
    page.rendered_html = "x" * (5 * 1024 * 1024 + 1)
    with pytest.raises(ListingExtractionError) as error:
        render_html_with_playwright(
            "https://example.com/start", resolver=public_resolver,
            playwright_factory=fake_playwright_factory(page),
        )
    assert error.value.code == "too_large"
    assert page.closed


def test_playwright_renderer_reports_missing_optional_dependency(monkeypatch):
    original_import = builtins.__import__

    def import_without_playwright(name, *args, **kwargs):
        if name == "playwright.sync_api":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_playwright)
    with pytest.raises(ListingExtractionError) as error:
        render_html_with_playwright(
            "https://example.com/start", resolver=public_resolver,
        )
    assert error.value.code == "browser_unavailable"


def test_playwright_renderer_closes_all_resources_after_render_failure(browser_http):
    page = FakePage()
    page.goto_error = RuntimeError("render failed")
    manager = FakePlaywrightManager(page)
    with pytest.raises(ListingExtractionError) as error:
        render_html_with_playwright(
            "https://example.com/start", resolver=public_resolver,
            playwright_factory=lambda: manager,
        )
    assert error.value.code == "browser_failed"
    assert page.closed is True
    assert manager.browser.context.closed is True
    assert manager.browser.closed is True


def test_playwright_renderer_closes_context_and_browser_after_page_creation_failure(browser_http):
    manager = FakePlaywrightManager(FakePage(), fail_new_page=True)
    with pytest.raises(ListingExtractionError) as error:
        render_html_with_playwright(
            "https://example.com/start", resolver=public_resolver,
            playwright_factory=lambda: manager,
        )
    assert error.value.code == "browser_failed"
    assert manager.browser.context.closed is True
    assert manager.browser.closed is True


def test_playwright_renderer_closes_browser_after_context_creation_failure(browser_http):
    manager = FakePlaywrightManager(FakePage(), fail_context=True)
    with pytest.raises(ListingExtractionError) as error:
        render_html_with_playwright(
            "https://example.com/start", resolver=public_resolver,
            playwright_factory=lambda: manager,
        )
    assert error.value.code == "browser_failed"
    assert manager.browser.closed is True


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


def image_bytes(color):
    buffer = BytesIO()
    Image.new("RGB", (32, 32), color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_download_images_validates_content_and_deduplicates_bytes(tmp_path):
    red = image_bytes("red")
    responses = {
        "https://images.example.com/a.jpg": FakeResponse(
            200, {"Content-Type": "image/jpeg"}, [red]
        ),
        "https://images.example.com/copy.jpg": FakeResponse(
            200, {"Content-Type": "image/jpeg"}, [red]
        ),
        "https://images.example.com/not-image.jpg": FakeResponse(
            200, {"Content-Type": "text/plain"}, [b"nope"]
        ),
    }
    paths, warnings = download_images(
        list(responses),
        tmp_path,
        http_get=lambda url, **kwargs: responses[url],
        resolver=public_resolver,
    )
    assert len(paths) == 1
    assert Path(paths[0]).is_file()
    assert warnings == ["Some listing images could not be used."]


def test_download_images_rejects_truncated_image_data(tmp_path):
    response = FakeResponse(
        200,
        {"Content-Type": "image/jpeg"},
        [image_bytes("red")[:-1]],
    )

    paths, warnings = download_images(
        ["https://images.example.com/truncated.jpg"],
        tmp_path,
        http_get=lambda *args, **kwargs: response,
        resolver=public_resolver,
    )

    assert paths == []
    assert list(tmp_path.iterdir()) == []
    assert warnings == ["Some listing images could not be used."]


def test_download_images_rejects_oversized_content_length(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.listing_images.MAX_IMAGE_BYTES", 8)
    response = FakeResponse(
        200,
        {"Content-Type": "image/jpeg", "Content-Length": "9"},
        [b"123456789"],
    )
    paths, warnings = download_images(
        ["https://images.example.com/truck.jpg"],
        tmp_path,
        http_get=lambda *args, **kwargs: response,
        resolver=public_resolver,
    )
    assert paths == []
    assert warnings == ["Some listing images could not be used."]


def test_download_images_rejects_private_host_before_http(tmp_path):
    requested = []

    def resolver(host, port, type=socket.SOCK_STREAM):
        return [(socket.AF_INET, type, 6, "", ("127.0.0.1", port))]

    paths, warnings = download_images(
        ["http://internal.example/truck.jpg"],
        tmp_path,
        http_get=lambda *args, **kwargs: requested.append(args[0]),
        resolver=resolver,
    )
    assert requested == []
    assert paths == []
    assert warnings == ["Some listing images could not be used."]


def test_download_images_revalidates_redirected_image_host(tmp_path):
    requested = []

    def resolver(host, port, type=socket.SOCK_STREAM):
        address = "127.0.0.1" if host == "127.0.0.1" else "93.184.216.34"
        return [(socket.AF_INET, type, 6, "", (address, port))]

    response = FakeResponse(302, {"Location": "http://127.0.0.1/private.jpg"})

    def get(url, **kwargs):
        requested.append(url)
        return response

    paths, _ = download_images(
        ["https://images.example.com/truck.jpg"],
        tmp_path,
        http_get=get,
        resolver=resolver,
    )
    assert requested == ["https://images.example.com/truck.jpg"]
    assert paths == []


def test_download_images_removes_partial_file_after_interrupted_write(
    monkeypatch, tmp_path
):
    def interrupted_write(path, data):
        with path.open("wb") as file:
            file.write(data[:16])
        raise OSError("write interrupted")

    monkeypatch.setattr(Path, "write_bytes", interrupted_write)
    response = FakeResponse(
        200,
        {"Content-Type": "image/jpeg"},
        [image_bytes("red")],
    )

    paths, warnings = download_images(
        ["https://images.example.com/truck.jpg"],
        tmp_path,
        http_get=lambda *args, **kwargs: response,
        resolver=public_resolver,
    )

    assert paths == []
    assert list(tmp_path.iterdir()) == []
    assert warnings == ["Some listing images could not be used."]


def test_download_images_keeps_at_most_eight_and_attempts_at_most_twenty_four(
    monkeypatch, tmp_path
):
    monkeypatch.setattr("pipeline.listing_images.MAX_IMAGES", 2)
    monkeypatch.setattr("pipeline.listing_images.MAX_CANDIDATES", 3)
    requested = []

    def get(url, **kwargs):
        requested.append(url)
        return FakeResponse(
            200,
            {"Content-Type": "image/jpeg"},
            [image_bytes((len(requested), 0, 0))],
        )

    paths, _ = download_images(
        [f"https://images.example.com/{i}.jpg" for i in range(10)],
        tmp_path,
        http_get=get,
        resolver=public_resolver,
    )
    assert len(paths) == 2
    assert len(requested) <= 3


def test_extract_listing_images_returns_only_image_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "pipeline.listing_images.fetch_html",
        lambda *args, **kwargs: (
            "https://example.com/truck",
            '<meta property="og:image" content="https://images.example.com/truck.jpg">',
        ),
    )
    monkeypatch.setattr(
        "pipeline.listing_images.download_images",
        lambda *args, **kwargs: ([str(tmp_path / "truck.jpg")], []),
    )
    result = extract_listing_images(
        "https://example.com/truck", tmp_path, resolver=public_resolver
    )
    assert result == {
        "source_url": "https://example.com/truck",
        "source_host": "example.com",
        "image_paths": [str(tmp_path / "truck.jpg")],
        "warnings": [],
    }
    assert set(result) == {"source_url", "source_host", "image_paths", "warnings"}


def test_extract_listing_images_uses_only_purple_wave_api_image_fields(tmp_path):
    listing_url = (
        "https://www.purplewave.com/auction/260917/item/FK3297/"
        "2017-Peterbilt-389-Trucks-Truck_Tractor-Kansas"
    )
    api_url = (
        "https://www.purplewave.com/v1/search/auction/260917/item/FK3297"
        "?return_fields=image_url,image_files"
    )
    image_base = "https://d323w7klwy72q3.cloudfront.net/i/a/2026/20260917truck"
    requested = []

    def get(url, **kwargs):
        requested.append(url)
        if url == listing_url:
            return FakeResponse(chunks=[b"<html></html>"])
        if url == api_url:
            return FakeResponse(
                headers={"Content-Type": "application/json"},
                chunks=[
                    (
                        '{"image_url":"' + image_base + '",'
                        '"image_files":["FK3297.JPG","FK3297A.JPG"],'
                        '"current_bid":"72000",'
                        '"price_image":"https://tracker.example/72000.jpg"}'
                    ).encode()
                ],
            )
        colors = {
            f"{image_base}/FK3297.JPG": "red",
            f"{image_base}/FK3297A.JPG": "blue",
        }
        return FakeResponse(
            headers={"Content-Type": "image/jpeg"},
            chunks=[image_bytes(colors[url])],
        )

    result = extract_listing_images(
        listing_url,
        tmp_path,
        http_get=get,
        resolver=public_resolver,
    )

    assert len(result["image_paths"]) == 2
    assert requested == [
        listing_url,
        api_url,
        f"{image_base}/FK3297.JPG",
        f"{image_base}/FK3297A.JPG",
    ]
    assert all("72000" not in url for url in requested)


def test_browser_failure_becomes_manual_upload_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "pipeline.listing_images.fetch_html",
        lambda *args, **kwargs: ("https://example.com/truck", "<html></html>"),
    )

    def fail_renderer(url):
        raise ListingExtractionError(
            "browser_failed", "Automatic extraction failed; upload photos manually."
        )

    with pytest.raises(ListingExtractionError) as error:
        extract_listing_images(
            "https://example.com/truck",
            tmp_path,
            browser_renderer=fail_renderer,
            resolver=public_resolver,
        )
    assert error.value.code == "browser_failed"


def test_extract_listing_images_uses_browser_only_when_static_extraction_has_no_candidates(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "pipeline.listing_images.fetch_html",
        lambda *args, **kwargs: ("https://example.com/truck", "<html></html>"),
    )
    monkeypatch.setattr(
        "pipeline.listing_images.download_images",
        lambda *args, **kwargs: ([str(tmp_path / "truck.jpg")], []),
    )
    rendered = []
    result = extract_listing_images(
        "https://example.com/truck",
        tmp_path,
        browser_renderer=lambda url: rendered.append(url)
        or (
            url,
            '<meta property="og:image" content="https://images.example.com/truck.jpg">',
        ),
        resolver=public_resolver,
    )
    assert rendered == ["https://example.com/truck"]
    assert result["image_paths"]


def test_extract_listing_images_raises_when_no_valid_images_remain(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "pipeline.listing_images.fetch_html",
        lambda *args, **kwargs: ("https://example.com/truck", "<html></html>"),
    )
    with pytest.raises(ListingExtractionError, match="photos"):
        extract_listing_images(
            "https://example.com/truck", tmp_path, resolver=public_resolver
        )


def test_fetch_html_revalidates_redirect_targets():
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


def test_fetch_html_rejects_fourth_redirect():
    responses = iter(
        FakeResponse(302, {"Location": f"/redirect-{index}"})
        for index in range(4)
    )
    with pytest.raises(ListingExtractionError) as error:
        fetch_html(
            "https://example.com/start",
            http_get=lambda *args, **kwargs: next(responses),
            resolver=public_resolver,
        )
    assert error.value.code == "redirect"


@pytest.mark.parametrize("status", [401, 403, 429])
def test_fetch_html_reports_access_blocks(status):
    with pytest.raises(ListingExtractionError) as error:
        fetch_html(
            "https://example.com/truck",
            http_get=lambda *args, **kwargs: FakeResponse(status),
            resolver=public_resolver,
        )
    assert error.value.code == "blocked"


def test_fetch_html_converts_request_timeout():
    def timeout(*args, **kwargs):
        raise requests.Timeout("slow")

    with pytest.raises(ListingExtractionError) as error:
        fetch_html(
            "https://example.com/truck",
            http_get=timeout,
            resolver=public_resolver,
        )
    assert error.value.code == "unavailable"


def test_fetch_html_rejects_oversized_body(monkeypatch):
    monkeypatch.setattr("pipeline.listing_images.MAX_HTML_BYTES", 8)
    response = FakeResponse(chunks=[b"123456", b"789"])

    with pytest.raises(ListingExtractionError, match="large"):
        fetch_html(
            "https://example.com/truck",
            http_get=lambda *args, **kwargs: response,
            resolver=public_resolver,
        )


def test_fetch_html_rejects_malformed_content_length():
    response = FakeResponse(headers={"Content-Type": "text/html", "Content-Length": "unknown"})

    with pytest.raises(ListingExtractionError, match="size"):
        fetch_html(
            "https://example.com/truck",
            http_get=lambda *args, **kwargs: response,
            resolver=public_resolver,
        )


def test_fetch_html_does_not_append_chunk_past_limit(monkeypatch):
    monkeypatch.setattr("pipeline.listing_images.MAX_HTML_BYTES", 8)
    response = FakeResponse(chunks=[b"123456789"])

    with pytest.raises(ListingExtractionError, match="large"):
        fetch_html(
            "https://example.com/truck",
            http_get=lambda *args, **kwargs: response,
            resolver=public_resolver,
        )


def test_fetch_html_returns_final_url_and_decoded_html():
    response = FakeResponse(chunks=[b"<html>truck</html>"])
    assert fetch_html(
        "https://example.com/truck",
        http_get=lambda *args, **kwargs: response,
        resolver=public_resolver,
    ) == ("https://example.com/truck", "<html>truck</html>")


@pytest.mark.parametrize("address", [
    "224.0.0.1", "239.255.255.250", "ff02::1", "ff0e::1", "fec0::1",
    "::ffff:224.0.0.1", "0.0.0.0", "240.0.0.1", "::", "100::1",
    "fe80::1", "::ffff:127.0.0.1",
])
def test_validate_public_url_rejects_prohibited_address_categories(address):
    def resolver(host, port, type=socket.SOCK_STREAM):
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        return [(family, type, 6, "", (address, port))]

    with pytest.raises(ListingExtractionError) as error:
        validate_public_url("https://example.com/truck", resolver=resolver)
    assert error.value.code == "unsafe_url"


def test_validate_public_url_normalizes_malformed_idna_error():
    with pytest.raises(ListingExtractionError) as error:
        validate_public_url("https://a..com/truck", resolver=public_resolver)
    assert error.value.code == "unsafe_url"


@pytest.mark.parametrize("markup", ["<meta property>", "<img srcset>", "<script type></script>"])
def test_malformed_attributes_do_not_hide_valid_images(markup):
    assert extract_image_urls(
        markup + '<img src="/truck.jpg">', "https://example.com/truck"
    ) == ["https://example.com/truck.jpg"]


def test_malformed_candidate_url_is_skipped():
    assert extract_image_urls(
        '<img src="http://[oops"><img src="/truck.jpg">',
        "https://www.purplewave.com/truck",
    ) == ["https://www.purplewave.com/truck.jpg"]


@pytest.mark.parametrize("failure", [
    requests.ConnectionError, requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ContentDecodingError, requests.Timeout,
])
def test_fetch_html_normalizes_stream_errors_and_closes_response(failure):
    class InterruptedResponse(FakeResponse):
        closed = False

        def iter_content(self, chunk_size):
            yield b"<html>"
            raise failure("interrupted response")

        def close(self):
            self.closed = True

    response = InterruptedResponse()
    with pytest.raises(ListingExtractionError) as error:
        fetch_html(
            "https://example.com/truck",
            http_get=lambda *args, **kwargs: response,
            resolver=public_resolver,
        )
    assert error.value.code == "unavailable"
    assert response.closed


@pytest.mark.parametrize("code", ["blocked", "unsafe_url", "too_large", "redirect", "unavailable", "unsupported"])
def test_static_fetch_failures_never_trigger_browser(code, tmp_path, monkeypatch):
    def fail_fetch(*args, **kwargs):
        raise ListingExtractionError(code, "Static fetch failed.")

    monkeypatch.setattr("pipeline.listing_images.fetch_html", fail_fetch)
    with pytest.raises(ListingExtractionError) as error:
        extract_listing_images(
            "https://example.com/truck", tmp_path, resolver=public_resolver,
            browser_renderer=lambda url: pytest.fail("browser must not run"),
        )
    assert error.value.code == code


@pytest.mark.parametrize("valid_from, expected_attempts, expected_retained", [(0, 8, 8), (23, 24, 1)])
def test_browser_candidates_use_bounded_downloader(
    valid_from, expected_attempts, expected_retained, tmp_path, monkeypatch
):
    page = FakePage()
    page.rendered_html = "".join(f'<img src="/image-{i}.jpg">' for i in range(30))
    requested_images = []

    def get(url, **kwargs):
        assert kwargs["stream"] is True
        assert kwargs["allow_redirects"] is False
        if url.endswith("/listing"):
            return FakeResponse(chunks=[b"<script>/* inline gallery */</script>"])
        requested_images.append(url)
        index = len(requested_images) - 1
        body = image_bytes((index * 8, 0, 0)) if index >= valid_from else b"invalid"
        return FakeResponse(headers={"Content-Type": "image/jpeg"}, chunks=[body])

    monkeypatch.setattr("pipeline.listing_images.requests.get", get)
    result = extract_listing_images(
        "https://example.com/listing", tmp_path, resolver=public_resolver,
        browser_renderer=lambda url: render_html_with_playwright(
            url, resolver=public_resolver, playwright_factory=fake_playwright_factory(page),
        ),
    )
    assert len(requested_images) == expected_attempts
    assert len(result["image_paths"]) == expected_retained
    assert all(Path(path).is_file() for path in result["image_paths"])
    assert set(result) == {"source_url", "source_host", "image_paths", "warnings"}


@pytest.mark.parametrize("declared", [True, False])
def test_browser_candidate_image_stops_at_fifteen_mib(declared, tmp_path, monkeypatch):
    page = FakePage()
    chunks_read = []

    def chunks():
        for index in range(300):
            chunks_read.append(index)
            yield b"x" * (64 * 1024)

    def get(url, **kwargs):
        if url.endswith("/listing"):
            return FakeResponse(chunks=[b"<html></html>"])
        headers = {"Content-Type": "image/jpeg"}
        if declared:
            headers["Content-Length"] = str(15 * 1024 * 1024 + 1)
        return FakeResponse(headers=headers, chunks=chunks())

    monkeypatch.setattr("pipeline.listing_images.requests.get", get)
    with pytest.raises(ListingExtractionError) as error:
        extract_listing_images(
            "https://example.com/listing", tmp_path, resolver=public_resolver,
            browser_renderer=lambda url: render_html_with_playwright(
                url, resolver=public_resolver, playwright_factory=fake_playwright_factory(page),
            ),
        )
    assert error.value.code == "no_images"
    assert len(chunks_read) == (0 if declared else 241)
    assert list(tmp_path.iterdir()) == []


def test_fetch_html_malformed_redirect_is_domain_error():
    with pytest.raises(ListingExtractionError) as error:
        fetch_html(
            "https://example.com/listing", resolver=public_resolver,
            http_get=lambda *args, **kwargs: FakeResponse(302, {"Location": "http://[oops"}),
        )
    assert error.value.code == "unsafe_url"


def test_download_images_skips_malformed_redirect(tmp_path):
    paths, warnings = download_images(
        ["https://example.com/image.jpg"], tmp_path, resolver=public_resolver,
        http_get=lambda *args, **kwargs: FakeResponse(302, {"Location": "http://[oops"}),
    )
    assert paths == []
    assert warnings == ["Some listing images could not be used."]
