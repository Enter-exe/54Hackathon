import json
from pathlib import Path

import pandas as pd


def write_listings_csv(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    output["image_paths"] = output["image_paths"].map(json.dumps)
    output.to_csv(path, index=False)


def read_listings_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"ad_id": str})
    frame["image_paths"] = frame["image_paths"].map(json.loads)
    return frame
