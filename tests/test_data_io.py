import pandas as pd

from pipeline.data_io import read_listings_csv, write_listings_csv


def test_listing_csv_round_trip_preserves_ids_and_image_lists(tmp_path):
    frame = pd.DataFrame({
        "ad_id": ["FB5018"],
        "price": [10_780.0],
        "image_paths": [["data/images/a.jpg", "data/images/b.jpg"]],
    })
    path = tmp_path / "listings_clean.csv"

    write_listings_csv(frame, path)
    loaded = read_listings_csv(path)

    assert loaded.loc[0, "ad_id"] == "FB5018"
    assert loaded.loc[0, "image_paths"] == ["data/images/a.jpg", "data/images/b.jpg"]
