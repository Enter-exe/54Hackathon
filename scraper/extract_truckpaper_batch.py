"""Parse a saved browser_batch tool-result JSON file (from scraping
TruckPaper) and append normalized rows to truckpaper_raw.jsonl, without
routing the whole payload through the agent's context window. Combines
what extract_from_batch_result.py + truckpaper_normalize.py do separately
for the CTT scrape, since TruckPaper's per-page payload is plain JSON
(no extra unescaping layer needed beyond the tool-result wrapper).
"""
import json
import sys

from truckpaper_normalize import normalize_row


def extract(path: str, out_jsonl: str) -> int:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    existing_ids = set()
    try:
        with open(out_jsonl, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_ids.add(json.loads(line)["ad_id"])
    except FileNotFoundError:
        pass

    n = 0
    with open(out_jsonl, "a", encoding="utf-8") as out:
        for item in data:
            text = item.get("text", "")
            if not text.startswith("[javascript_tool:javascript_exec]"):
                continue
            quoted = text.split("]", 1)[1].strip()
            try:
                arr_json_str, _ = json.JSONDecoder().raw_decode(quoted)
                rows = json.loads(arr_json_str)
            except Exception as e:
                print(f"WARN: failed to parse chunk: {e}", file=sys.stderr)
                continue
            for row in rows:
                normalized = normalize_row(row)
                if normalized["price"] is None or not normalized["photo_urls"]:
                    continue
                if normalized["ad_id"] in existing_ids:
                    continue
                existing_ids.add(normalized["ad_id"])
                out.write(json.dumps(normalized) + "\n")
                n += 1
    print(f"Extracted {n} new rows from {path}")
    return n


if __name__ == "__main__":
    extract(sys.argv[1], sys.argv[2])
