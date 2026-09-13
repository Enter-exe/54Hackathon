"""Scrape Commercial Truck Trader box-truck listings.

CTT's search-results page is a client-rendered Vue SPA behind DataDome bot
protection: a plain HTTP request gets blocked (HTTP 202, empty body), so
listing data is instead pulled from `window.searchData.search.results`
after a real browser renders the page. Sliced by make (facet) and paginated
per make, because a single query's pagination caps out at 10 pages (420
listings) even when more exist across the whole category. The image CDN
(cdn-media.tilabs.io) sits outside DataDome and is fetched separately with
plain `requests`.
"""
import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from playwright.sync_api import sync_playwright

BASE_URL = "https://www.commercialtrucktrader.com/Box/trucks-for-sale"
PAGE_SIZE = 42
CDN_TEMPLATE = (
    "https://cdn-media.tilabs.io/v1/media/{photo_id}.webp"
    "?width=800&height=600&quality=85&bestfit=true"
)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

FIELDS = [
    "ad_id", "year", "make_name", "model_name", "price", "mileage",
    "city", "state_code", "zip_code", "class_name", "category_name",
    "condition", "photo_count", "create_date", "ad_detail_url",
]


def _raw(v):
    return v.get("raw") if isinstance(v, dict) else v


def extract_row(result: dict) -> dict:
    row = {f: _raw(result.get(f)) for f in FIELDS}
    row["photo_ids"] = _raw(result.get("photo_ids")) or []
    if isinstance(row["category_name"], list):
        row["category_name"] = ";".join(row["category_name"])
    if isinstance(row["make_name"], list):
        row["make_name"] = row["make_name"][0] if row["make_name"] else None
    if isinstance(row["model_name"], list):
        row["model_name"] = row["model_name"][0] if row["model_name"] else None
    return row


def fetch_search_data(page, url: str, retries: int = 3):
    for attempt in range(retries):
        try:
            page.goto(url, wait_until="networkidle", timeout=30000)
        except Exception as e:
            print(f"WARN: navigation error on {url}: {e}")
            time.sleep(2 + attempt)
            continue
        data = page.evaluate("() => window.searchData || null")
        if data and data.get("search", {}).get("results"):
            return data
        time.sleep(2 + attempt)
    return None


def scrape_listings(headless: bool, delay_range: tuple, max_makes: int | None) -> pd.DataFrame:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless, channel="chrome")
        context = browser.new_context(user_agent=USER_AGENT, viewport={"width": 1366, "height": 900})
        page = context.new_page()

        base_data = fetch_search_data(page, f"{BASE_URL}?upfitCategory=Box%20Trucks")
        if base_data is None:
            browser.close()
            raise RuntimeError(
                "Could not load the base search page - site structure may have "
                "changed, or the request was blocked. Try headless=False."
            )

        makes = []
        for facet_group in base_data["search"]["facets"].get("make_facet", []):
            if facet_group.get("name") == "all_makes":
                for entry in facet_group["data"]:
                    name, make_id = entry["value"].split("|")
                    makes.append((name, make_id, entry["count"]))
        if max_makes:
            makes = makes[:max_makes]

        rows = []
        seen_ids = set()
        for name, make_id, count in makes:
            num_pages = min(10, -(-count // PAGE_SIZE))  # ceil, capped at site's 10-page limit
            for p in range(1, num_pages + 1):
                url = f"{BASE_URL}?upfitCategory=Box%20Trucks&make={quote(name)}%7C{make_id}&page={p}"
                data = fetch_search_data(page, url)
                if data is None:
                    print(f"WARN: failed to load {name} page {p}, skipping")
                    continue
                for result in data["search"]["results"]:
                    row = extract_row(result)
                    if row["ad_id"] in seen_ids:
                        continue
                    seen_ids.add(row["ad_id"])
                    rows.append(row)
                time.sleep(random.uniform(*delay_range))
            print(f"{name}: {count} listed on site, {len(seen_ids)} total collected so far")

        browser.close()
    return pd.DataFrame(rows)


def _download_one(ad_id, photo_id, idx, out_dir: Path) -> bool:
    # CTT's photo_ids are bare hex ids used with CDN_TEMPLATE; other sources
    # (e.g. TruckPaper) store the already-resolved full URL directly instead.
    # Always save as .webp regardless of actual format -- the rest of the
    # pipeline globs specifically for *.webp, and PIL detects format from
    # file content, not extension, so this is safe.
    is_full_url = str(photo_id).startswith("http")
    dest = out_dir / str(ad_id) / f"{idx}_{'url' if is_full_url else photo_id}.webp"
    if dest.exists() and dest.stat().st_size > 2048:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = photo_id if is_full_url else CDN_TEMPLATE.format(photo_id=photo_id)
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        if resp.status_code == 200 and len(resp.content) > 2048:
            dest.write_bytes(resp.content)
            return True
    except requests.RequestException:
        pass
    return False


def download_images(df: pd.DataFrame, out_dir: Path, max_images_per_listing: int, workers: int = 16):
    jobs = []
    for _, row in df.iterrows():
        for idx, photo_id in enumerate(row["photo_ids"][:max_images_per_listing]):
            jobs.append((row["ad_id"], photo_id, idx))

    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_download_one, ad_id, photo_id, idx, out_dir) for ad_id, photo_id, idx in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            if fut.result():
                ok += 1
            if i % 200 == 0:
                print(f"  downloaded {i}/{len(jobs)} images ({ok} ok)")
    print(f"Image download done: {ok}/{len(jobs)} succeeded")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="data", help="root output directory")
    ap.add_argument("--headless", action="store_true", help="run browser headless (riskier vs DataDome)")
    ap.add_argument("--max-images-per-listing", type=int, default=6)
    ap.add_argument("--skip-images", action="store_true", help="only scrape metadata, skip image download")
    ap.add_argument("--max-makes", type=int, default=None, help="limit number of make-facets (for a quick test run)")
    ap.add_argument("--min-delay", type=float, default=1.5)
    ap.add_argument("--max-delay", type=float, default=3.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    images_dir = out_dir / "images"
    raw_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    print("Scraping listing metadata from Commercial Truck Trader...")
    df = scrape_listings(
        headless=args.headless,
        delay_range=(args.min_delay, args.max_delay),
        max_makes=args.max_makes,
    )
    print(f"Collected {len(df)} unique listings")

    listings_path = raw_dir / "listings.parquet"
    df.to_parquet(listings_path, index=False)
    print(f"Wrote {listings_path}")

    if not args.skip_images and len(df):
        print("Downloading images (CDN, not behind DataDome)...")
        download_images(df, images_dir, args.max_images_per_listing)


if __name__ == "__main__":
    main()
