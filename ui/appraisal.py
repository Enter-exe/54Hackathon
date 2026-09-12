"""Core appraisal pipeline for the demo UI, kept separate from ui/app.py's
Streamlit rendering so it can be tested/run directly without a browser.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import torch

from modeling.confidence import adjust_price_range
from modeling.train_price import apply_margin, predict_quantiles
from pipeline.condition_assessment import ATTRIBUTES, assess_images, load_text_embeddings
from pipeline.extract_embeddings import embed_images, load_model
from pipeline.gate_images import gate_images

PRICE_MODEL_PATH = Path("artifacts/price_model/price_models.joblib")
CALIBRATION_PATH = Path("data/processed/condition_calibration.json")


def load_resources(price_model_path: Path = PRICE_MODEL_PATH, calibration_path: Path = CALIBRATION_PATH):
    if not Path(price_model_path).exists():
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_model(device)
    condition_text_pairs = load_text_embeddings(model, device)
    bundle = joblib.load(price_model_path)
    calibration = json.loads(Path(calibration_path).read_text()) if Path(calibration_path).exists() else None
    return {
        "device": device,
        "model": model,
        "preprocess": preprocess,
        "condition_text_pairs": condition_text_pairs,
        "bundle": bundle,
        "calibration": calibration,
    }


def run_appraisal(paths: list[str], resources: dict) -> dict:
    """Runs the full pipeline on a listing's photos: gate -> embed -> condition
    -> price. Returns {"gate": ..., "condition": ... | None, "price": ... | None}.
    condition/price are None when the gate rejects all photos.
    """
    device, model, preprocess = resources["device"], resources["model"], resources["preprocess"]
    gate_result = gate_images(paths, model, preprocess, device)

    if not gate_result["accepted"]:
        return {"gate": gate_result, "condition": None, "price": None}

    usable_paths = gate_result["usable_paths"]
    embs = embed_images(usable_paths, model, preprocess, device)
    pooled = embs.mean(axis=0)
    pooled = pooled / (np.linalg.norm(pooled) + 1e-8)

    thresholds = resources["calibration"]["thresholds"] if resources["calibration"] else None
    condition_result = assess_images(
        usable_paths, model, preprocess, device, resources["condition_text_pairs"], thresholds=thresholds
    )

    condition_features = np.array(
        [condition_result[attr["name"]]["probability"] for attr in ATTRIBUTES], dtype=np.float32
    )
    X = np.concatenate([pooled, condition_features]).reshape(1, -1).astype(np.float32)

    bundle = resources["bundle"]
    raw_quantiles = predict_quantiles(bundle["models"], X)
    calibrated_quantiles = apply_margin(raw_quantiles, bundle["calibration_margin"])[0]
    price_result = adjust_price_range(calibrated_quantiles, gate_result)

    return {"gate": gate_result, "condition": condition_result, "price": price_result}
