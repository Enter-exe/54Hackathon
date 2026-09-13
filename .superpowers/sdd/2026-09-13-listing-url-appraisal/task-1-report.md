# Task 1 report

## Files changed

- `pipeline/listing_images.py`: added `ListingExtractionError`, public HTTP(S) URL validation, bounded HTML streaming, manual redirect handling, content-type checks, and response decoding.
- `tests/test_listing_images.py`: added URL-shape, public-address, fragment-normalization, redirect revalidation, body-size, and HTML-fetch tests.

## Tests run

- `python -m pytest tests/test_listing_images.py -v` — **12 passed in 0.12s**.
- `git diff --check` — passed.
- `python -m py_compile pipeline/listing_images.py tests/test_listing_images.py` — passed.
- `python -m pytest -q` — collection blocked by pre-existing missing dependencies: `torch`, `sklearn`, `open_clip`, and `joblib` (9 collection errors; no regression tests executed).

## Design decisions

- Only `http` and `https` URLs using standard ports are accepted; credentials and fragments are removed from normalized URLs.
- DNS results are checked with `ipaddress.ip_address(...).is_global`, and every resolved address must be public.
- Redirects are followed manually for at most three hops, with every target revalidated before fetching.
- Responses stream in 64 KiB chunks and are capped at 5 MiB; non-HTML, blocked, unavailable, and oversized responses receive stable error codes.

## Concerns

- The complete repository suite cannot currently run in this worktree because its ML/training dependencies are not installed. The focused Task 1 suite and syntax checks pass.

## Review follow-up

- Added a deterministic malformed `Content-Length` test and convert invalid declarations to `ListingExtractionError(code="too_large")`.
- Changed streaming enforcement to check `len(body) + len(chunk)` before extending the body, so the in-memory buffer never exceeds the configured maximum.
- `python -m pytest tests/test_listing_images.py -v` — **14 passed in 0.06s**.
- `python -m py_compile pipeline/listing_images.py tests/test_listing_images.py` — passed.
- `git diff --check` — passed.
- `python -m pytest -q` — still blocked during collection by the same 9 missing-dependency errors (`torch`, `sklearn`, `open_clip`, and `joblib`).
