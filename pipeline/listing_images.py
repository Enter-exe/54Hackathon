import ipaddress
import socket
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests


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
