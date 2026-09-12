"""Parse a saved browser_batch tool-result JSON file and append the listing
rows produced by its javascript_exec steps to a JSONL file. Used to pull
listing data collected via a real (non-automated-looking) browser session
into the project's dataset without routing the whole payload through the
agent's context window.
"""
import json
import sys


def extract(path: str, out_jsonl: str) -> int:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    n = 0
    with open(out_jsonl, "a", encoding="utf-8") as out:
        for item in data:
            text = item.get("text", "")
            if not text.startswith("[javascript_tool:javascript_exec]"):
                continue
            quoted = text.split("]", 1)[1].strip()
            try:
                arr_json_str, _ = json.JSONDecoder().raw_decode(quoted)
                arr = json.loads(arr_json_str)
            except Exception as e:
                print(f"WARN: failed to parse chunk: {e}", file=sys.stderr)
                continue
            for row in arr:
                out.write(json.dumps(row) + "\n")
                n += 1
    print(f"Extracted {n} rows from {path}")
    return n


if __name__ == "__main__":
    extract(sys.argv[1], sys.argv[2])
