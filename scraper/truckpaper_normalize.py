"""Normalize raw TruckPaper listing rows (scraped via the browser, see
scraper/append_page_json.py for how raw pages get here) into the same
canonical schema as the CTT scrape, so both sources can be merged.

Key differences from CTT that this accounts for:
  - TruckPaper's search cards expose a full, already-resolved CDN image URL
    per listing (no separate photo-id + template), and only ONE photo is
    reachable without visiting each listing's own detail page (its other
    photos are lazy-loaded per-listing, not worth the extra page-visit cost
    at this volume) -- so every TruckPaper listing contributes exactly 1
    photo, versus up to 4 for CTT listings.
  - No GVWR weight class field is exposed on the search card (only on the
    detail page, not fetched here) -- class_name is left null for these
    rows. pipeline/train_spec_classifier.py's load_data() drops null-label
    rows before training, so this doesn't corrupt the weight-class
    classifier, it just means TruckPaper rows don't contribute to it.
  - Location is "City, Full State Name" rather than CTT's "city" + 2-letter
    state_code -- state is kept as the full name rather than guessing an
    abbreviation mapping.
"""
import json
import re
import sys

STATE_ABBREV = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV", "new hampshire": "NH",
    "new jersey": "NJ", "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN",
    "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}


def parse_price(price_text):
    if not price_text:
        return None
    if "call" in price_text.lower():
        return None
    match = re.search(r"[\d,]+", price_text)
    return int(match.group(0).replace(",", "")) if match else None


def parse_mileage(mileage_text):
    if not mileage_text:
        return None
    match = re.search(r"[\d,]+", mileage_text)
    return int(match.group(0).replace(",", "")) if match else None


def parse_title(title):
    if not title:
        return None, None, None
    match = re.match(r"^(\d{4})\s+(\S+)\s+(.*)$", title.strip())
    if not match:
        return None, None, None
    year, make, model = match.groups()
    return int(year), make.upper(), model.strip()


def parse_location(location):
    if not location or "," not in location:
        return location, None
    city, state_name = [s.strip() for s in location.rsplit(",", 1)]
    state_code = STATE_ABBREV.get(state_name.lower(), state_name)
    return city.lower(), state_code


def hires_image_url(url):
    return re.sub(r"w=\d+&h=\d+", "w=800&h=600", url) if url else url


def normalize_row(row):
    year, make, model = parse_title(row.get("title"))
    city, state_code = parse_location(row.get("location"))
    return {
        "ad_id": int(row["listing_id"]),
        "year": year,
        "make_name": make,
        "model_name": model,
        "price": parse_price(row.get("price_text")),
        "mileage": parse_mileage(row.get("mileage")),
        "city": city,
        "state_code": state_code,
        "class_name": None,
        "category_name": row.get("category"),
        "condition": None,
        "photo_count": 1 if row.get("image_url") else 0,
        "photo_urls": [hires_image_url(row["image_url"])] if row.get("image_url") else [],
        "source": "truckpaper",
    }


def main():
    src, dest = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8") as f:
        rows = json.load(f)

    n = 0
    with open(dest, "a", encoding="utf-8") as out:
        for row in rows:
            normalized = normalize_row(row)
            if normalized["price"] is None or not normalized["photo_urls"]:
                continue
            out.write(json.dumps(normalized) + "\n")
            n += 1
    print(f"Normalized and wrote {n}/{len(rows)} rows from {src} (dropped: no price or no photo)")


if __name__ == "__main__":
    main()
