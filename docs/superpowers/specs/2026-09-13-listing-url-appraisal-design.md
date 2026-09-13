# Listing URL Appraisal Design

**Date:** 2026-09-13
**Status:** Approved in chat

## Goal

Let a user paste a public truck-listing URL into the Streamlit app, extract
the listing photos, and run the existing image-first appraisal without using
the page's asking price, current bid, or other price fields as model inputs.
Manual photo upload remains available when automatic extraction is blocked or
unsupported.

## Scope

The first version supports three extraction layers:

1. Dedicated parsing for Purple Wave and Commercial Truck Trader.
2. Generic extraction from JSON-LD, OpenGraph metadata, and HTML image tags.
3. An optional Playwright renderer for JavaScript-only pages when a browser is
   installed in the runtime.

The feature targets conventional, publicly accessible HTTP(S) listing pages.
It will not bypass authentication, CAPTCHAs, paywalls, bot protection, or site
access controls. Those cases fall back to manual upload. A free-roaming AI
browsing agent is deliberately excluded: deterministic extraction is faster,
safer, cheaper, and testable, while an agent would still be unable to bypass
access controls reliably.

## User Experience

The existing upload area gains two clearly labeled paths:

- **Listing link:** paste a URL and choose **Extract photos and appraise**.
- **Upload photos:** keep the current multi-file uploader unchanged.

Only one path is appraised at a time. For a listing link, the app displays a
loading state while it fetches and validates images, then reports how many
usable photos it found and shows their thumbnails before the appraisal
results. If extraction fails, an inline message explains the reason and
directs the user to the upload control without losing the entered URL.

The UI never displays a scraped asking price or bid as part of the estimate.
It may identify the source hostname so the user can confirm which page was
processed.

## Architecture

### Extraction module

Add `pipeline/listing_images.py` with a small public API:

```python
extract_listing_images(url, output_dir, *, http_get=None, browser_renderer=None) -> dict
```

The returned dictionary contains only extraction metadata and local image
paths:

```python
{
    "source_url": "https://example.com/listing/123",
    "source_host": "example.com",
    "image_paths": ["/tmp/.../0.jpg", ...],
    "warnings": [],
}
```

The module does not return page price, bid, description, or vehicle metadata.
This narrow contract makes it impossible for those fields to enter
`run_appraisal()` accidentally.

The extraction sequence is:

1. Validate the page URL.
2. Fetch bounded HTML while validating every redirect.
3. Select the matching site adapter, then fall back to generic parsing.
4. If static HTML yields no candidates and a renderer is available, render
   once and repeat parsing against the resulting HTML.
5. Resolve relative image URLs, preserve page order, and deduplicate URLs.
6. Download bounded image responses into the supplied temporary directory.
7. Verify image content with Pillow, deduplicate identical bytes, and return
   at most eight valid local images.

Adapters and generic parsing remain pure functions over HTML wherever
possible, so they can be tested without live network access. Standard-library
`html.parser`, `json`, and `urllib.parse` are sufficient; the implementation
will reuse the repository's existing `requests` and Pillow dependencies.

### Appraisal integration

`ui/app.py` owns the temporary directory and passes only the extracted local
paths into the existing `run_appraisal(paths, resources)` function. The
existing upload path follows the same boundary.

Remote listing pages include logos, avatars, and recommendations, so the
current image gate must filter each candidate independently. The existing
behavior accepts every clear image when any one image looks like a truck;
that behavior will be tightened so only candidates meeting the per-image
truck threshold reach embedding, condition, and price inference.

### Browser fallback

Playwright is imported lazily. The extraction module accepts a renderer
callable instead of creating a browser during ordinary static extraction.
The production renderer will attempt one bounded page load and return rendered
HTML. If Playwright or Chromium is unavailable, the extractor returns a
specific warning and the UI offers manual upload. Browser installation is not
a prerequisite for the static and site-adapter paths.

## Security and Resource Limits

Because the app performs server-side requests to user-provided URLs, it must
guard against server-side request forgery and resource exhaustion:

- Accept only `http` and `https` URLs without embedded credentials.
- Restrict page URLs to standard HTTP(S) ports.
- Resolve hostnames and reject loopback, private, link-local, multicast,
  reserved, and otherwise non-global IP addresses, including IPv6.
- Revalidate every redirect target and every candidate image URL.
- Limit redirects to three and use explicit connection/read timeouts.
- Limit HTML to 5 MB, each image to 15 MB, candidate downloads to 24, and
  retained images to eight.
- Require an image content type and successful Pillow verification.
- Static parsing never evaluates scripts found in fetched HTML. The optional
  browser may execute page JavaScript only inside a fresh context with no
  authentication state and with downloads disabled.
- Never interpret page content as instructions or allow it to alter the
  extraction and appraisal workflow.
- Do not send authentication cookies or attempt to defeat access controls.

DNS validation reduces the practical SSRF surface for this prototype but is
not a complete defense against DNS rebinding. A production deployment should
put outbound fetching behind an egress proxy that blocks private address
ranges at connection time.

## Error Handling

Expected failures are returned as concise domain errors rather than raw
network exceptions:

- Invalid or unsafe URL.
- Page unavailable, redirected unsafely, too large, or timed out.
- Authentication, CAPTCHA, or bot protection encountered.
- Unsupported page with no discoverable listing images.
- Candidate images unavailable, oversized, or malformed.
- Optional browser renderer unavailable.

Partially successful extraction continues with the valid images and surfaces
warnings. Zero valid images always routes the user to manual upload. Temporary
files are deleted after the Streamlit rerun completes.

## Testing

Automated tests will use local HTML fixtures and injected HTTP responses; the
normal suite will not depend on changing external websites.

- URL validation rejects unsafe schemes, credentials, ports, private IPs,
  unsafe redirects, and unsafe image hosts.
- Purple Wave, Commercial Truck Trader, JSON-LD, OpenGraph, `src`, `data-src`,
  and `srcset` fixtures produce ordered, deduplicated image candidates.
- Response-size, timeout, redirect, candidate-count, image-size, MIME, Pillow,
  and byte-deduplication limits are enforced.
- Browser fallback runs only after static extraction fails and degrades
  cleanly when unavailable.
- UI integration passes only extracted local image paths to `run_appraisal`.
- Per-image gating excludes unrelated page images even when another image is
  a valid truck photo.
- Existing upload and appraisal tests remain green.

Optional live smoke tests may exercise supported public pages manually, but
they will not be part of the deterministic test suite.

## Success Criteria

- Pasting the tested Purple Wave listing URL extracts truck photos and returns
  an appraisal without consuming its current bid.
- A conventional page exposing listing images through structured metadata is
  handled without a site-specific adapter.
- Blocked or unsupported pages fall back to manual upload without crashing.
- Unsafe URLs and oversized responses are rejected before appraisal.
- Existing manual upload behavior remains unchanged.
