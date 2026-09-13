# Listing URL Appraisal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users paste a public truck-listing URL, extract and validate its truck photos, and run the existing image-only appraisal without using the listing's price or bid.

**Architecture:** Add one focused extraction module that validates public URLs, parses site-specific and generic image metadata, downloads bounded image files, and optionally renders JavaScript pages with Playwright. Connect its narrow image-path-only result to the existing Streamlit appraisal flow, and tighten the CLIP gate so unrelated page images are rejected individually.

**Tech Stack:** Python 3, Streamlit, requests, Pillow, standard-library `html.parser`/`ipaddress`/`socket`/`urllib.parse`, optional Playwright, pytest

**Spec:** `docs/superpowers/specs/2026-09-13-listing-url-appraisal-design.md`

## Global Constraints

- Accept only public `http` and `https` URLs without credentials and with standard ports.
- Revalidate every redirect and image URL; reject every resolved IP that is not globally routable.
- Limit redirects to 3, HTML to 5 MB, each image to 15 MB, candidate downloads to 24, and retained images to 8.
- Return only source identity, local image paths, and warnings from extraction; never return page price, bid, description, or vehicle metadata.
- Do not bypass authentication, CAPTCHAs, paywalls, bot protection, or access controls.
- Browser rendering is optional and must degrade to manual upload when unavailable.
- Static parsing never evaluates page scripts; the optional browser uses a fresh context with no authentication state and with downloads disabled.
- Preserve the existing manual-upload appraisal flow.
- Use no new HTML-parsing dependency; standard-library parsing is sufficient.

---

### Task 1: Safe URL validation and bounded HTML fetching

**Files:**
- Create: `pipeline/listing_images.py`
- Create: `tests/test_listing_images.py`

**Interfaces:**
- Consumes: user URL strings, an optional `resolver(host, port, type=socket.SOCK_STREAM)`, and an optional requests-compatible `http_get` callable.
- Produces: `ListingExtractionError(code: str, message: str)`, `validate_public_url(url, resolver=socket.getaddrinfo) -> str`, and `fetch_html(url, *, http_get=None, resolver=socket.getaddrinfo) -> tuple[str, str]` where the tuple is `(final_url, html)`.

- [ ] **Step 1: Write failing URL-validation tests**

```python
import socket

import pytest

from pipeline.listing_images import ListingExtractionError, validate_public_url


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
```

- [ ] **Step 2: Run the URL tests to verify RED**

Run: `python -m pytest tests/test_listing_images.py -v`

Expected: FAIL during collection because `pipeline.listing_images` does not exist.

- [ ] **Step 3: Implement the URL validator**

```python
# pipeline/listing_images.py
import ipaddress
import socket
from urllib.parse import urlsplit, urlunsplit


class ListingExtractionError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def validate_public_url(url, resolver=socket.getaddrinfo):
    try:
        parts = urlsplit(str(url).strip())
        port = parts.port
    except ValueError as exc:
        raise ListingExtractionError("unsafe_url", "The listing URL is unsafe or malformed.") from exc

    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ListingExtractionError("unsafe_url", "The listing URL must use public HTTP or HTTPS.")
    if parts.username is not None or parts.password is not None:
        raise ListingExtractionError("unsafe_url", "The listing URL cannot contain credentials.")
    expected_port = 443 if parts.scheme == "https" else 80
    if port not in {None, expected_port}:
        raise ListingExtractionError("unsafe_url", "The listing URL must use a standard HTTP or HTTPS port.")

    host = parts.hostname.encode("idna").decode("ascii")
    try:
        addresses = resolver(host, port or expected_port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ListingExtractionError("unavailable", "The listing hostname could not be resolved.") from exc
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ListingExtractionError("unsafe_url", "The listing hostname must resolve only to public addresses.")

    display_host = f"[{host}]" if ":" in host else host
    netloc = display_host if port is None else f"{display_host}:{port}"
    return urlunsplit((parts.scheme, netloc, parts.path or "/", parts.query, ""))
```

- [ ] **Step 4: Run the URL tests to verify GREEN**

Run: `python -m pytest tests/test_listing_images.py -v`

Expected: all URL-validation tests PASS.

- [ ] **Step 5: Write failing bounded-fetch tests**

```python
from pipeline.listing_images import fetch_html


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
    monkeypatch.setattr("pipeline.listing_images.MAX_HTML_BYTES", 8)
    response = FakeResponse(chunks=[b"123456", b"789"])

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
```

- [ ] **Step 6: Run the fetch tests to verify RED**

Run: `python -m pytest tests/test_listing_images.py -v`

Expected: FAIL because `fetch_html` is not defined.

- [ ] **Step 7: Implement manual redirects and bounded streaming**

```python
# add to pipeline/listing_images.py
from urllib.parse import urljoin

import requests

MAX_REDIRECTS = 3
MAX_HTML_BYTES = 5 * 1024 * 1024
TIMEOUT = (5, 10)
USER_AGENT = "Kamion image-only appraisal/1.0"


def _read_bounded(response, maximum):
    declared = response.headers.get("Content-Length")
    if declared and int(declared) > maximum:
        raise ListingExtractionError("too_large", "The remote response is too large.")
    body = bytearray()
    for chunk in response.iter_content(64 * 1024):
        body.extend(chunk)
        if len(body) > maximum:
            raise ListingExtractionError("too_large", "The remote response is too large.")
    return bytes(body)


def fetch_html(url, *, http_get=None, resolver=socket.getaddrinfo):
    get = http_get or requests.get
    current = validate_public_url(url, resolver)
    for redirect_count in range(MAX_REDIRECTS + 1):
        try:
            response = get(
                current,
                headers={"User-Agent": USER_AGENT},
                timeout=TIMEOUT,
                allow_redirects=False,
                stream=True,
            )
        except requests.RequestException as exc:
            raise ListingExtractionError("unavailable", "The listing page could not be fetched.") from exc
        try:
            if response.status_code in {301, 302, 303, 307, 308}:
                if redirect_count == MAX_REDIRECTS or not response.headers.get("Location"):
                    raise ListingExtractionError("redirect", "The listing redirected too many times.")
                current = validate_public_url(urljoin(current, response.headers["Location"]), resolver)
                continue
            if response.status_code in {401, 403, 429}:
                raise ListingExtractionError("blocked", "The listing site blocked automatic access.")
            if response.status_code != 200:
                raise ListingExtractionError("unavailable", "The listing page is unavailable.")
            if "html" not in response.headers.get("Content-Type", "").lower():
                raise ListingExtractionError("unsupported", "The URL did not return an HTML listing page.")
            return current, _read_bounded(response, MAX_HTML_BYTES).decode(
                response.encoding or "utf-8", errors="replace"
            )
        finally:
            response.close()
    raise AssertionError("redirect loop must return or raise")
```

- [ ] **Step 8: Run Task 1 tests and commit**

Run: `python -m pytest tests/test_listing_images.py -v`

Expected: PASS.

```bash
git add pipeline/listing_images.py tests/test_listing_images.py
git commit -m "feat: validate and fetch listing URLs safely"
```

---

### Task 2: Site-specific and generic image discovery

**Files:**
- Modify: `pipeline/listing_images.py`
- Modify: `tests/test_listing_images.py`

**Interfaces:**
- Consumes: fetched HTML and its final public page URL.
- Produces: `extract_image_urls(html: str, page_url: str) -> list[str]`, ordered with site-specific listing images first and duplicates removed.

- [ ] **Step 1: Write failing parser tests using local HTML fixtures**

```python
from pipeline.listing_images import extract_image_urls


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
```

- [ ] **Step 2: Run parser tests to verify RED**

Run: `python -m pytest tests/test_listing_images.py -k extract -v`

Expected: FAIL because `extract_image_urls` is not defined.

- [ ] **Step 3: Implement one standard-library parser and adapter ordering**

```python
# add to pipeline/listing_images.py
import json
from html.parser import HTMLParser


class _ImageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []
        self.json_scripts = []
        self._json_parts = None

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "meta" and values.get("property", values.get("name", "")).lower() in {
            "og:image", "og:image:url", "twitter:image"
        }:
            self.urls.append(values.get("content"))
        if tag == "img":
            self.urls.extend(values.get(name) for name in ("src", "data-src", "data-lazy-src"))
            for item in values.get("srcset", "").split(","):
                self.urls.append(item.strip().split(" ", 1)[0])
        if tag == "script" and values.get("type", "").lower() == "application/ld+json":
            self._json_parts = []

    def handle_data(self, data):
        if self._json_parts is not None:
            self._json_parts.append(data)

    def handle_endtag(self, tag):
        if tag == "script" and self._json_parts is not None:
            self.json_scripts.append("".join(self._json_parts))
            self._json_parts = None


def _json_image_values(value, key=None):
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from _json_image_values(child, child_key.lower())
    elif isinstance(value, list):
        for child in value:
            yield from _json_image_values(child, key)
    elif isinstance(value, str) and key in {"image", "contenturl", "thumbnailurl"}:
        yield value


def _dedupe(values):
    return list(dict.fromkeys(value for value in values if value))


def extract_image_urls(html, page_url):
    parser = _ImageParser()
    parser.feed(html)
    structured = []
    for script in parser.json_scripts:
        try:
            structured.extend(_json_image_values(json.loads(script)))
        except (json.JSONDecodeError, TypeError):
            continue
    absolute = [
        urljoin(page_url, value)
        for value in [*structured, *parser.urls]
        if isinstance(value, str) and value and not value.startswith("data:")
    ]
    host = urlsplit(page_url).hostname or ""
    if host.endswith("purplewave.com"):
        preferred = [url for url in absolute if "cloudfront.net/i/a/" in url]
    elif host.endswith("commercialtrucktrader.com"):
        preferred = [url for url in absolute if "tilabs.io" in url]
    else:
        preferred = []
    return _dedupe([*preferred, *absolute])
```

- [ ] **Step 4: Run parser tests and the complete extraction test file**

Run: `python -m pytest tests/test_listing_images.py -v`

Expected: PASS.

- [ ] **Step 5: Commit image discovery**

```bash
git add pipeline/listing_images.py tests/test_listing_images.py
git commit -m "feat: discover listing images from structured HTML"
```

---

### Task 3: Bounded image downloads and extraction orchestration

**Files:**
- Modify: `pipeline/listing_images.py`
- Modify: `tests/test_listing_images.py`

**Interfaces:**
- Consumes: validated candidate image URLs, output directory, injected HTTP getter, and optional `browser_renderer(url) -> tuple[str, str]`.
- Produces: `download_images(urls, output_dir, *, http_get=None, resolver=socket.getaddrinfo) -> tuple[list[str], list[str]]` and `extract_listing_images(url, output_dir, *, http_get=None, browser_renderer=None, resolver=socket.getaddrinfo) -> dict`.

- [ ] **Step 1: Write failing image-validation and deduplication tests**

```python
from io import BytesIO
from pathlib import Path

from PIL import Image

from pipeline.listing_images import download_images


def image_bytes(color):
    buffer = BytesIO()
    Image.new("RGB", (32, 32), color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_download_images_validates_content_and_deduplicates_bytes(tmp_path):
    red = image_bytes("red")
    responses = {
        "https://images.example.com/a.jpg": FakeResponse(200, {"Content-Type": "image/jpeg"}, [red]),
        "https://images.example.com/copy.jpg": FakeResponse(200, {"Content-Type": "image/jpeg"}, [red]),
        "https://images.example.com/not-image.jpg": FakeResponse(200, {"Content-Type": "text/plain"}, [b"nope"]),
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


def test_download_images_keeps_at_most_eight_and_attempts_at_most_twenty_four(monkeypatch, tmp_path):
    monkeypatch.setattr("pipeline.listing_images.MAX_IMAGES", 2)
    monkeypatch.setattr("pipeline.listing_images.MAX_CANDIDATES", 3)
    requested = []

    def get(url, **kwargs):
        requested.append(url)
        return FakeResponse(200, {"Content-Type": "image/jpeg"}, [image_bytes(url[-5])])

    paths, _ = download_images(
        [f"https://images.example.com/{i}.jpg" for i in range(10)],
        tmp_path,
        http_get=get,
        resolver=public_resolver,
    )
    assert len(paths) == 2
    assert len(requested) <= 3
```

- [ ] **Step 2: Run download tests to verify RED**

Run: `python -m pytest tests/test_listing_images.py -k download -v`

Expected: FAIL because `download_images` is not defined.

- [ ] **Step 3: Implement bounded image download and byte deduplication**

```python
# add to pipeline/listing_images.py
import hashlib
from io import BytesIO
from pathlib import Path

from PIL import Image

MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_CANDIDATES = 24
MAX_IMAGES = 8


def download_images(urls, output_dir, *, http_get=None, resolver=socket.getaddrinfo):
    get = http_get or requests.get
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths, hashes = [], set()
    rejected = 0
    for image_url in urls[:MAX_CANDIDATES]:
        if len(paths) == MAX_IMAGES:
            break
        try:
            current = validate_public_url(image_url, resolver)
            for redirect_count in range(MAX_REDIRECTS + 1):
                response = get(
                    current,
                    headers={"User-Agent": USER_AGENT},
                    timeout=TIMEOUT,
                    allow_redirects=False,
                    stream=True,
                )
                try:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        if redirect_count == MAX_REDIRECTS or not response.headers.get("Location"):
                            raise ListingExtractionError("redirect", "An image redirected too many times.")
                        current = validate_public_url(
                            urljoin(current, response.headers["Location"]), resolver
                        )
                        continue
                    if response.status_code != 200 or not response.headers.get("Content-Type", "").lower().startswith("image/"):
                        raise ListingExtractionError("invalid_image", "A candidate URL did not return an image.")
                    content = _read_bounded(response, MAX_IMAGE_BYTES)
                    break
                finally:
                    response.close()
            with Image.open(BytesIO(content)) as image:
                image.verify()
                suffix = "." + (image.format or "jpg").lower().replace("jpeg", "jpg")
            digest = hashlib.sha256(content).digest()
            if digest in hashes:
                rejected += 1
                continue
            hashes.add(digest)
            path = output_dir / f"{len(paths)}{suffix}"
            path.write_bytes(content)
            paths.append(str(path))
        except (ListingExtractionError, OSError, requests.RequestException):
            rejected += 1
    warnings = ["Some listing images could not be used."] if rejected else []
    return paths, warnings
```

- [ ] **Step 4: Run download tests to verify GREEN**

Run: `python -m pytest tests/test_listing_images.py -k download -v`

Expected: PASS.

- [ ] **Step 5: Write failing orchestration and narrow-contract tests**

```python
from pipeline.listing_images import extract_listing_images


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
    result = extract_listing_images("https://example.com/truck", tmp_path, resolver=public_resolver)
    assert result == {
        "source_url": "https://example.com/truck",
        "source_host": "example.com",
        "image_paths": [str(tmp_path / "truck.jpg")],
        "warnings": [],
    }
    assert set(result) == {"source_url", "source_host", "image_paths", "warnings"}


def test_extract_listing_images_uses_browser_only_when_static_extraction_has_no_candidates(tmp_path, monkeypatch):
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
        browser_renderer=lambda url: rendered.append(url) or (
            url,
            '<meta property="og:image" content="https://images.example.com/truck.jpg">',
        ),
        resolver=public_resolver,
    )
    assert rendered == ["https://example.com/truck"]
    assert result["image_paths"]


def test_extract_listing_images_raises_when_no_valid_images_remain(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "pipeline.listing_images.fetch_html",
        lambda *args, **kwargs: ("https://example.com/truck", "<html></html>"),
    )
    with pytest.raises(ListingExtractionError, match="photos"):
        extract_listing_images("https://example.com/truck", tmp_path, resolver=public_resolver)
```

- [ ] **Step 6: Run orchestration tests to verify RED**

Run: `python -m pytest tests/test_listing_images.py -k "contract or browser or remain" -v`

Expected: FAIL because `extract_listing_images` is not defined.

- [ ] **Step 7: Implement extraction orchestration**

```python
# add to pipeline/listing_images.py
def extract_listing_images(
    url,
    output_dir,
    *,
    http_get=None,
    browser_renderer=None,
    resolver=socket.getaddrinfo,
):
    safe_url = validate_public_url(url, resolver)
    warnings = []
    used_browser = False
    try:
        final_url, html = fetch_html(safe_url, http_get=http_get, resolver=resolver)
        candidates = extract_image_urls(html, final_url)
    except ListingExtractionError as exc:
        if browser_renderer is None:
            raise
        warnings.append(f"Static extraction failed: {exc}")
        final_url, html = browser_renderer(safe_url)
        used_browser = True
        final_url = validate_public_url(final_url, resolver)
        candidates = extract_image_urls(html, final_url)

    if not candidates and browser_renderer is not None and not used_browser:
        final_url, html = browser_renderer(final_url)
        final_url = validate_public_url(final_url, resolver)
        candidates = extract_image_urls(html, final_url)
    if not candidates:
        raise ListingExtractionError("no_images", "No listing photos could be discovered; upload them manually.")

    paths, download_warnings = download_images(
        candidates, output_dir, http_get=http_get, resolver=resolver
    )
    if not paths:
        raise ListingExtractionError("no_images", "No valid listing photos could be downloaded; upload them manually.")
    return {
        "source_url": final_url,
        "source_host": urlsplit(final_url).hostname,
        "image_paths": paths,
        "warnings": [*warnings, *download_warnings],
    }
```

- [ ] **Step 8: Run the complete extraction test file and commit**

Run: `python -m pytest tests/test_listing_images.py -v`

Expected: PASS.

```bash
git add pipeline/listing_images.py tests/test_listing_images.py
git commit -m "feat: download validated listing photos"
```

---

### Task 4: Optional Playwright renderer

**Files:**
- Modify: `pipeline/listing_images.py`
- Modify: `tests/test_listing_images.py`

**Interfaces:**
- Consumes: a validated public URL and optional injected Playwright factory.
- Produces: `render_html_with_playwright(url, *, resolver=socket.getaddrinfo, playwright_factory=None) -> tuple[str, str]`.

- [ ] **Step 1: Write failing renderer tests**

```python
from types import SimpleNamespace

from pipeline.listing_images import render_html_with_playwright


class FakePage:
    url = "https://example.com/final"

    def route(self, pattern, handler):
        self.route_handler = handler

    def goto(self, url, wait_until, timeout):
        self.requested_url = url

    def wait_for_load_state(self, state, timeout):
        pass

    def content(self):
        return '<meta property="og:image" content="/truck.jpg">'


class FakeBrowser:
    def __init__(self, page):
        self.page = page

    def new_context(self, user_agent, accept_downloads):
        assert accept_downloads is False
        return SimpleNamespace(new_page=lambda: self.page)

    def close(self):
        pass


class FakePlaywrightManager:
    def __init__(self, page):
        browser = FakeBrowser(page)
        self.playwright = SimpleNamespace(
            chromium=SimpleNamespace(launch=lambda headless: browser)
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


def test_playwright_renderer_returns_final_url_and_html():
    page = FakePage()
    factory = fake_playwright_factory(page)
    final_url, html = render_html_with_playwright(
        "https://example.com/start",
        resolver=public_resolver,
        playwright_factory=factory,
    )
    assert final_url == "https://example.com/final"
    assert "truck.jpg" in html


def test_playwright_renderer_rejects_unsafe_final_navigation():
    page = FakePage()
    page.url = "http://127.0.0.1/private"
    with pytest.raises(ListingExtractionError, match="public"):
        render_html_with_playwright(
            "https://example.com/start",
            resolver=public_resolver_with_localhost,
            playwright_factory=fake_playwright_factory(page),
        )
```

- [ ] **Step 2: Run renderer tests to verify RED**

Run: `python -m pytest tests/test_listing_images.py -k playwright -v`

Expected: FAIL because `render_html_with_playwright` is not defined.

- [ ] **Step 3: Implement lazy, bounded browser rendering**

```python
# add to pipeline/listing_images.py
BROWSER_TIMEOUT_MS = 10_000


def render_html_with_playwright(
    url,
    *,
    resolver=socket.getaddrinfo,
    playwright_factory=None,
):
    safe_url = validate_public_url(url, resolver)
    if playwright_factory is None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise ListingExtractionError(
                "browser_unavailable",
                "This page needs a browser to extract photos; upload them manually.",
            ) from exc
        playwright_factory = sync_playwright

    try:
        with playwright_factory() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                context = browser.new_context(
                    user_agent=USER_AGENT,
                    accept_downloads=False,
                )
                page = context.new_page()

                def guard_route(route):
                    try:
                        validate_public_url(route.request.url, resolver)
                    except ListingExtractionError:
                        route.abort()
                    else:
                        route.continue_()

                page.route("**/*", guard_route)
                page.goto(safe_url, wait_until="domcontentloaded", timeout=BROWSER_TIMEOUT_MS)
                try:
                    page.wait_for_load_state("networkidle", timeout=BROWSER_TIMEOUT_MS)
                except Exception:
                    pass
                final_url = validate_public_url(page.url, resolver)
                html = page.content()
                if len(html.encode("utf-8")) > MAX_HTML_BYTES:
                    raise ListingExtractionError("too_large", "The rendered listing page is too large.")
                return final_url, html
            finally:
                browser.close()
    except ListingExtractionError:
        raise
    except Exception as exc:
        raise ListingExtractionError(
            "browser_failed",
            "The listing could not be rendered automatically; upload its photos manually.",
        ) from exc
```

- [ ] **Step 4: Run renderer tests and all extraction tests**

Run: `python -m pytest tests/test_listing_images.py -v`

Expected: PASS, including clean fallback behavior without a real browser.

- [ ] **Step 5: Commit browser fallback**

```bash
git add pipeline/listing_images.py tests/test_listing_images.py
git commit -m "feat: add optional browser listing extraction"
```

---

### Task 5: Per-image truck filtering

**Files:**
- Modify: `pipeline/gate_images.py:62-109`
- Modify: `tests/test_gate_images.py:36-55`

**Interfaces:**
- Consumes: one CLIP truck-similarity margin per quality-approved candidate.
- Produces: `gate_images(paths, model, preprocess, device, dark_threshold, blur_threshold, truck_margin)` with only individually approved truck photos in `usable_paths`; each rejected non-truck candidate includes `reasons: ["not_truck"]` and its margin.

- [ ] **Step 1: Change the existing test to require individual filtering**

```python
def test_gate_filters_each_nontruck_photo_individually(monkeypatch, tmp_path):
    good = save_image(tmp_path / "good.png", checkerboard())
    nontruck = save_image(tmp_path / "nontruck.png", 255 - checkerboard())
    monkeypatch.setattr(
        "pipeline.gate_images.truck_similarity_margins",
        lambda *args: np.array([0.05, -0.01]),
    )

    result = gate_images([good, nontruck], object(), object(), "cpu")

    assert result["accepted"] is True
    assert result["usable_paths"] == [good]
    assert result["confidence"] == "low"
    assert result["rejected"][-1]["path"] == nontruck
    assert result["rejected"][-1]["reasons"] == ["not_truck"]
```

- [ ] **Step 2: Run the gate test to verify RED**

Run: `python -m pytest tests/test_gate_images.py::test_gate_filters_each_nontruck_photo_individually -v`

Expected: FAIL because the current gate keeps both candidates whenever one passes.

- [ ] **Step 3: Implement per-image margin filtering**

```python
# replace the aggregate max check in gate_images()
if candidates:
    margins = truck_similarity_margins(candidates, model, preprocess, device)
    if len(margins) != len(candidates):
        raise ValueError("CLIP returned the wrong number of image scores")
    for path, margin in zip(candidates, margins):
        if margin >= truck_margin:
            usable_paths.append(path)
        else:
            rejected.append(
                {
                    "path": path,
                    "brightness": None,
                    "sharpness": None,
                    "truck_margin": float(margin),
                    "reasons": ["not_truck"],
                }
            )
```

- [ ] **Step 4: Run the gate and appraisal tests**

Run: `python -m pytest tests/test_gate_images.py tests/test_ui_appraisal.py -v`

Expected: PASS, with artifact-dependent tests skipped only when local artifacts are absent.

- [ ] **Step 5: Commit the gate correction**

```bash
git add pipeline/gate_images.py tests/test_gate_images.py
git commit -m "fix: filter listing photos individually"
```

---

### Task 6: Streamlit listing-link flow

**Files:**
- Modify: `ui/app.py`
- Create: `tests/test_ui_app.py`

**Interfaces:**
- Consumes: a listing URL, temporary directory, loaded model resources, `extractor(url, output_dir, browser_renderer) -> dict`, and `appraiser(paths, resources) -> dict`.
- Produces: `appraise_listing_url(url, output_dir, resources, *, extractor=extract_listing_images, appraiser=run_appraisal, browser_renderer=render_html_with_playwright) -> tuple[dict, dict]` and a Streamlit choice between link extraction and manual upload.

- [ ] **Step 1: Write a failing boundary test proving only image paths reach appraisal**

```python
from ui.app import appraise_listing_url


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

    extraction, appraisal = appraise_listing_url(
        "https://example.com/truck",
        tmp_path,
        {"model": object()},
        extractor=extractor,
        appraiser=appraiser,
        browser_renderer=None,
    )

    assert observed["paths"] == ["/tmp/a.jpg", "/tmp/b.jpg"]
    assert set(extraction) == {"source_url", "source_host", "image_paths", "warnings"}
    assert appraisal["price"]["price_median"] == 12_000
```

- [ ] **Step 2: Run the boundary test to verify RED**

Run: `python -m pytest tests/test_ui_app.py -v`

Expected: FAIL because `appraise_listing_url` is not defined.

- [ ] **Step 3: Implement the testable UI boundary**

```python
# add imports in ui/app.py
from pipeline.listing_images import (
    ListingExtractionError,
    extract_listing_images,
    render_html_with_playwright,
)


def appraise_listing_url(
    url,
    output_dir,
    resources,
    *,
    extractor=extract_listing_images,
    appraiser=run_appraisal,
    browser_renderer=render_html_with_playwright,
):
    extraction = extractor(
        url,
        output_dir,
        browser_renderer=browser_renderer,
    )
    return extraction, appraiser(extraction["image_paths"], resources)
```

- [ ] **Step 4: Run the boundary test to verify GREEN**

Run: `python -m pytest tests/test_ui_app.py -v`

Expected: PASS.

- [ ] **Step 5: Add the mutually exclusive Streamlit input paths**

Replace the single uploader block in `main()` with a horizontal radio labeled
`How would you like to provide the truck photos?` and options `Listing link`
and `Upload photos`. The link path uses a form so editing the URL does not
trigger network requests:

```python
source_mode = st.radio(
    "How would you like to provide the truck photos?",
    ["Listing link", "Upload photos"],
    horizontal=True,
)

if source_mode == "Listing link":
    with st.form("listing-link-form"):
        listing_url = st.text_input(
            "Truck listing URL",
            placeholder="https://www.purplewave.com/auction/260917/item/FK3297",
        )
        submitted = st.form_submit_button("Extract photos and appraise", type="primary")
    if not submitted:
        st.caption("The estimate uses listing photos only—not the asking price or current bid.")
        return
    temporary = tempfile.TemporaryDirectory()
    try:
        with st.spinner("Extracting and analyzing listing photos…"):
            extraction, result = appraise_listing_url(
                listing_url, Path(temporary.name), resources
            )
        st.success(
            f"Extracted {len(extraction['image_paths'])} photos from {extraction['source_host']}."
        )
        for warning in extraction["warnings"]:
            st.warning(warning)
        st.image(extraction["image_paths"], width=150)
    except ListingExtractionError as exc:
        st.error(str(exc))
        st.info("Switch to Upload photos to continue manually.")
        return
    finally:
        temporary.cleanup()
else:
    uploaded_files = st.file_uploader(
        "Upload truck photos", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True
    )
    if not uploaded_files:
        st.caption("Bad lighting, awkward angles, a missing view—that's fine, upload what you have.")
        return
    with tempfile.TemporaryDirectory() as temporary:
        with st.spinner("Analyzing photos…"):
            paths = save_uploads(uploaded_files, Path(temporary))
            result = run_appraisal(paths, resources)
        st.image(list(uploaded_files), width=150)

if not result["gate"]["accepted"]:
    render_rejection(result["gate"])
    return
render_price(result["price"])
render_predicted_specs(result["predicted_class"], result["predicted_make"])
render_condition(result["condition"])
render_comparables(result["comparables"])
```

Remove the old unconditional uploader/inference block after moving its behavior
into the `Upload photos` branch. Change the page caption from “used box truck”
to “used commercial truck” because the model contains box trucks and tractors.

- [ ] **Step 6: Run UI and appraisal tests**

Run: `python -m pytest tests/test_ui_app.py tests/test_ui_appraisal.py tests/test_gate_images.py -v`

Expected: PASS, with artifact-dependent tests skipped only when artifacts are absent.

- [ ] **Step 7: Commit the UI flow**

```bash
git add ui/app.py tests/test_ui_app.py
git commit -m "feat: appraise truck photos from listing links"
```

---

### Task 7: Deployment dependencies, documentation, and end-to-end verification

**Files:**
- Modify: `ui/requirements.txt`
- Modify: `README.md`
- Modify: `tests/test_listing_images.py`

**Interfaces:**
- Consumes: the complete URL extraction and Streamlit flow.
- Produces: a deployable dependency declaration, user-facing run instructions, and deterministic regression coverage for all specified limits.

- [ ] **Step 1: Add missing limit/error regression tests**

Extend `tests/test_listing_images.py` with these explicit cases:

```python
import requests


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


def test_malformed_json_ld_does_not_hide_valid_open_graph_image():
    html = """
      <script type="application/ld+json">{not-json</script>
      <meta property="og:image" content="/truck.jpg">
    """
    assert extract_image_urls(html, "https://example.com/listing") == [
        "https://example.com/truck.jpg"
    ]


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
```

No test may access the internet.

- [ ] **Step 2: Run the new regression cases to verify RED where behavior is incomplete**

Run: `python -m pytest tests/test_listing_images.py -v`

Expected: any missing edge behavior FAILS for the named reason, not because of fixture errors.

- [ ] **Step 3: Make the smallest extraction changes needed for GREEN**

Keep all changes inside `pipeline/listing_images.py`. Do not add retries,
concurrency, caching, an adapter framework, an LLM, or a database. Exact error
codes remain `unsafe_url`, `unavailable`, `blocked`, `redirect`, `too_large`,
`unsupported`, `invalid_image`, `browser_unavailable`, `browser_failed`, and
`no_images`.

- [ ] **Step 4: Declare runtime dependencies used by the deployed UI**

Replace `ui/requirements.txt` with one package per line:

```text
streamlit
requests
Pillow
joblib
lightgbm
numpy
pandas
scikit-learn
torch
open_clip_torch
```

Do not make Playwright mandatory. Static extraction and the two adapters run
without it; environments that install `scraper/requirements.txt` gain the
optional browser path, and other deployments show manual upload.

- [ ] **Step 5: Document both input modes and their limits**

Add a `Running the demo` section to `README.md` with:

```markdown
## Running the demo

Install the UI/model dependencies and start Streamlit from the repository root:

```bash
python -m pip install -r ui/requirements.txt
streamlit run ui/app.py
```

The app accepts either uploaded truck photos or a public listing URL. URL
appraisal extracts photos only; asking prices and current bids are not model
inputs. Purple Wave and Commercial Truck Trader receive dedicated parsing,
with generic structured-image extraction for other public listing pages.
Pages blocked by authentication, CAPTCHA, or bot protection fall back to
manual photo upload. Run the data/model pipeline first when generated
artifacts are not already present.
```

- [ ] **Step 6: Run deterministic verification**

Run: `python -m pytest -q`

Expected: all tests PASS; only the existing artifact-dependent tests may SKIP.

Run: `python -m compileall -q pipeline ui tests`

Expected: exit 0 with no output.

Run: `git diff --check`

Expected: exit 0 with no whitespace errors.

- [ ] **Step 7: Run an optional live Purple Wave smoke check**

With network access explicitly permitted, run the extractor against the tested
public listing URL and a temporary directory:

```bash
python -c "from pathlib import Path; from tempfile import TemporaryDirectory; from pipeline.listing_images import extract_listing_images; d=TemporaryDirectory(); r=extract_listing_images('https://www.purplewave.com/auction/260917/item/FK3297/2017-Peterbilt-389-Trucks-Truck_Tractor-Kansas', Path(d.name)); print(r['source_host'], len(r['image_paths']), sorted(r))"
```

Expected: host `www.purplewave.com`, between 1 and 8 valid images, and exactly
the keys `image_paths`, `source_host`, `source_url`, and `warnings`. The command
must not print or return the current bid.

- [ ] **Step 8: Run interactive UI verification**

Start `streamlit run ui/app.py` with local model artifacts available. In the
browser, verify:

1. Listing-link mode has a visible URL label, submit button, and image-only note.
2. The Purple Wave smoke URL displays extracted thumbnails and an appraisal.
3. A private URL such as `http://127.0.0.1:8504` is rejected without a request.
4. An unsupported/blocked page displays the manual-upload fallback.
5. Upload mode still accepts several local images and renders its appraisal.
6. Desktop and narrow mobile-width screenshots have no clipped controls,
   horizontal overflow, or unreadable error text.

- [ ] **Step 9: Review the final diff and commit**

Run: `git status --short`, `git diff --stat`, and `git diff`.

Confirm every hunk traces to listing-link extraction, safe fetching, individual
image filtering, UI integration, deployment requirements, or documentation.

```bash
git add pipeline/listing_images.py tests/test_listing_images.py ui/requirements.txt README.md
git commit -m "docs: document listing URL appraisal"
```
