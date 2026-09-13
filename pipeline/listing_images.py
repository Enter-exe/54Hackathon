import hashlib
from io import BytesIO
import json
from html.parser import HTMLParser
import ipaddress
from pathlib import Path
import socket
from urllib.parse import urljoin, urlsplit, urlunsplit

from PIL import Image
import requests


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
        media_type = values.get("type", "").split(";", 1)[0].strip().lower()
        if tag == "script" and media_type == "application/ld+json":
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


def _canonical_image_url(value, page_url):
    if not isinstance(value, str) or not value:
        return None
    if urlsplit(value).scheme.lower() == "data":
        return None
    resolved = urljoin(page_url, value)
    parts = urlsplit(resolved)
    if parts.scheme.lower() == "data":
        return None
    return urlunsplit((parts.scheme.lower(), parts.netloc, parts.path, parts.query, ""))


def _host_is(host, domain):
    return host == domain or host.endswith("." + domain)


def extract_image_urls(html: str, page_url: str) -> list[str]:
    parser = _ImageParser()
    parser.feed(html)
    structured = []
    for script in parser.json_scripts:
        try:
            structured.extend(_json_image_values(json.loads(script)))
        except (json.JSONDecodeError, TypeError):
            continue
    absolute = [
        url
        for value in [*structured, *parser.urls]
        if (url := _canonical_image_url(value, page_url)) is not None
    ]
    host = (urlsplit(page_url).hostname or "").lower()
    if _host_is(host, "purplewave.com"):
        preferred = [
            url for url in absolute
            if _host_is((urlsplit(url).hostname or "").lower(), "cloudfront.net")
            and urlsplit(url).path.startswith("/i/a/")
        ]
    elif _host_is(host, "commercialtrucktrader.com"):
        preferred = [
            url for url in absolute
            if _host_is((urlsplit(url).hostname or "").lower(), "tilabs.io")
        ]
    else:
        preferred = []
    return _dedupe([*preferred, *absolute])


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


MAX_REDIRECTS = 3
MAX_HTML_BYTES = 5 * 1024 * 1024
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_CANDIDATES = 24
MAX_IMAGES = 8
TIMEOUT = (5, 10)
USER_AGENT = "Kamion image-only appraisal/1.0"
BROWSER_TIMEOUT_MS = 10_000


def _read_bounded(response, maximum):
    declared = response.headers.get("Content-Length")
    if declared:
        try:
            declared_size = int(declared)
        except (TypeError, ValueError) as exc:
            raise ListingExtractionError("too_large", "The remote response size is invalid.") from exc
        if declared_size > maximum:
            raise ListingExtractionError("too_large", "The remote response is too large.")
    body = bytearray()
    for chunk in response.iter_content(64 * 1024):
        if len(body) + len(chunk) > maximum:
            raise ListingExtractionError("too_large", "The remote response is too large.")
        body.extend(chunk)
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
                page.goto(
                    safe_url,
                    wait_until="domcontentloaded",
                    timeout=BROWSER_TIMEOUT_MS,
                )
                try:
                    page.wait_for_load_state("networkidle", timeout=BROWSER_TIMEOUT_MS)
                except Exception:
                    pass
                final_url = validate_public_url(page.url, resolver)
                html = page.content()
                if len(html.encode("utf-8")) > MAX_HTML_BYTES:
                    raise ListingExtractionError(
                        "too_large", "The rendered listing page is too large."
                    )
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
                        if redirect_count == MAX_REDIRECTS or not response.headers.get(
                            "Location"
                        ):
                            raise ListingExtractionError(
                                "redirect", "An image redirected too many times."
                            )
                        current = validate_public_url(
                            urljoin(current, response.headers["Location"]), resolver
                        )
                        continue
                    if response.status_code != 200 or not response.headers.get(
                        "Content-Type", ""
                    ).lower().startswith("image/"):
                        raise ListingExtractionError(
                            "invalid_image", "A candidate URL did not return an image."
                        )
                    content = _read_bounded(response, MAX_IMAGE_BYTES)
                    break
                finally:
                    response.close()
            with Image.open(BytesIO(content)) as image:
                image.load()
                suffix = "." + (image.format or "jpg").lower().replace("jpeg", "jpg")
            digest = hashlib.sha256(content).digest()
            if digest in hashes:
                rejected += 1
                continue
            hashes.add(digest)
            path = output_dir / f"{len(paths)}{suffix}"
            try:
                path.write_bytes(content)
            except OSError:
                path.unlink(missing_ok=True)
                raise
            paths.append(str(path))
        except (ListingExtractionError, OSError, requests.RequestException):
            rejected += 1
    warnings = ["Some listing images could not be used."] if rejected else []
    return paths, warnings


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
        raise ListingExtractionError(
            "no_images", "No listing photos could be discovered; upload them manually."
        )

    paths, download_warnings = download_images(
        candidates, output_dir, http_get=http_get, resolver=resolver
    )
    if not paths:
        raise ListingExtractionError(
            "no_images",
            "No valid listing photos could be downloaded; upload them manually.",
        )
    return {
        "source_url": final_url,
        "source_host": urlsplit(final_url).hostname,
        "image_paths": paths,
        "warnings": [*warnings, *download_warnings],
    }
