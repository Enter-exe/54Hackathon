from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).parents[1] / "app.py"


def _render_result(result, photo_names=None):
    from app import render_result

    render_result(result, photo_names)


def test_app_renders_upload_flow_without_loading_models():
    # Catches model loading during initial render or removing the demo's core flow.
    app = AppTest.from_file(APP_PATH).run(timeout=10)

    assert not app.exception
    assert app.title[0].value == "Kamion Truck Appraisal"
    assert app.file_uploader[0].label == "Truck photos"
    assert app.button[0].label == "Appraise truck"
    assert "USD estimates" in app.caption[0].value
    assert "100 completed US auction sales" in app.caption[0].value
    assert "European or Turkish markets" in app.caption[0].value


def test_save_uploads_uses_numeric_filenames_and_safe_extensions(tmp_path):
    # Catches writing a user-controlled upload name into the temporary directory.
    from app import save_uploads

    uploads = [
        SimpleNamespace(name="../../cab.JPG", getbuffer=lambda: b"first"),
        SimpleNamespace(name="details.webp", getbuffer=lambda: b"second"),
    ]

    paths = save_uploads(uploads, tmp_path)

    assert paths == [str(tmp_path / "1.jpg"), str(tmp_path / "2.webp")]
    assert [Path(path).read_bytes() for path in paths] == [b"first", b"second"]


def test_render_result_shows_complete_accepted_appraisal(tmp_path):
    # Catches dropping price, confidence, condition, or comparable details.
    thumbnail = tmp_path / "truck.png"
    thumbnail.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDAT\x08\xd7c\xf8\xcf"
        b"\xc0\xf0\x1f\x00\x05\x00\x01\xff\x89\x99=\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    result = {
        "accepted": True,
        "price_low": 22_500.0,
        "price_median": 30_000.0,
        "price_high": 38_750.0,
        "confidence": "high",
        "warnings": ["limited_photo_coverage"],
        "reasons": [],
        "condition": {
            "rust": {"probability": 0.1, "flag": False},
            "body_damage": {"probability": 0.8, "flag": True},
        },
        "comparables": [
            {
                "ad_id": str(index),
                "similarity": 0.91 - index / 100,
                "price": 20_000.0 + index,
                "year": 2022,
                "make_name": "Freight*[liner]",
                "model_name": "Cascadia_(sleeper)",
                "source_url": f"https://example.com/{index}",
                "thumbnail_path": str(thumbnail),
            }
            for index in range(3)
        ],
    }

    app = AppTest.from_function(
        _render_result, args=(result,), default_timeout=10
    ).run()

    assert not app.exception
    assert [(metric.label, metric.value) for metric in app.metric[:3]] == [
        ("Low", "$22,500"),
        ("Market estimate", "$30,000"),
        ("High", "$38,750"),
    ]
    assert app.success[0].value == "High confidence"
    assert app.warning[0].value.startswith("Limited photo coverage")
    condition_values = {
        "**Rust**",
        "10% signal",
        "No issue flagged",
        "**Body damage**",
        "80% signal",
        "**Review**",
    }
    assert [
        item.value for item in app.markdown if item.value in condition_values
    ] == [
        "**Rust**",
        "10% signal",
        "No issue flagged",
        "**Body damage**",
        "80% signal",
        "**Review**",
    ]
    assert len(app.image) == 3
    assert [item.value for item in app.text] == [
        "2022 Freight*[liner] Cascadia_(sleeper)",
    ] * 3
    assert [(metric.label, metric.value) for metric in app.metric[3:]] == [
        ("Sale price", "$20,000"),
        ("Sale price", "$20,001"),
        ("Sale price", "$20,002"),
    ]
    assert [caption.value for caption in app.caption[-3:]] == [
        "91% visual similarity",
        "90% visual similarity",
        "89% visual similarity",
    ]
    assert [button.proto.url for button in app.get("link_button")] == [
        "https://example.com/0",
        "https://example.com/1",
        "https://example.com/2",
    ]


def test_render_result_shows_rejection_recovery_by_photo():
    # Catches dropping engine rejections or pairing them with the wrong upload name.
    from pipeline.appraise import AppraisalEngine

    result = AppraisalEngine._rejected({
        "accepted": False,
        "confidence": "low",
        "warnings": ["some_photos_rejected"],
        "reasons": ["no_usable_truck_photos"],
        "rejected": [
            {"path": "/tmp/2.png", "reasons": ["too_dark", "too_blurry"]},
            {"path": "/tmp/1.jpg", "reasons": ["not_truck"]},
        ],
    })

    app = AppTest.from_function(
        _render_result,
        args=(result, ["*[cab](https://invalid.example)*.jpg", "side.png"]),
        default_timeout=10,
    ).run()

    assert not app.exception
    assert app.error[0].value.startswith("No usable truck photos")
    assert not app.metric
    assert [item.value for item in app.text] == [
        "side.png — Too dark, Too blurry",
        "*[cab](https://invalid.example)*.jpg — Not a truck",
    ]
