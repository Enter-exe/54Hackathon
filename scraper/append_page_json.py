"""Append a JSON array file (one page's worth of listings, captured via the
browser) to the scraped_raw.jsonl dataset, deduping by ad_id."""
import json
import sys
from pathlib import Path


def main():
    src, dest = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8") as f:
        rows = json.load(f)

    existing_ids = set()
    dest_path = Path(dest)
    if dest_path.exists():
        with open(dest_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_ids.add(json.loads(line)["ad_id"])

    n = 0
    with open(dest_path, "a", encoding="utf-8") as out:
        for row in rows:
            if row["ad_id"] in existing_ids:
                continue
            out.write(json.dumps(row) + "\n")
            existing_ids.add(row["ad_id"])
            n += 1
    print(f"Appended {n} new rows ({len(rows) - n} duplicates skipped) from {src}")


if __name__ == "__main__":
    main()
