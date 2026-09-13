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
from modeling.train_price import predict_range
from pipeline.condition_assessment import ATTRIBUTES, assess_images, load_text_embeddings
from pipeline.extract_embeddings import embed_images, load_model
from pipeline.gate_images import gate_images

PRICE_MODEL_PATH = Path("artifacts/price_model/price_models.joblib")
CALIBRATION_PATH = Path("data/processed/condition_calibration.json")
COMPARABLES_INDEX_PATH = Path("data/processed/comparables_index.npz")
MAKE_CLASSIFIER_PATH = Path("artifacts/spec_classifier/make_classifier.joblib")
CLASS_CLASSIFIER_PATH = Path("artifacts/spec_classifier/class_classifier.joblib")


def load_resources(
    price_model_path: Path = PRICE_MODEL_PATH,
    calibration_path: Path = CALIBRATION_PATH,
    comparables_index_path: Path = COMPARABLES_INDEX_PATH,
    make_classifier_path: Path = MAKE_CLASSIFIER_PATH,
    class_classifier_path: Path = CLASS_CLASSIFIER_PATH,
):
    if not Path(price_model_path).exists():
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_model(device)
    condition_text_pairs = load_text_embeddings(model, device)
    bundle = joblib.load(price_model_path)
    calibration = json.loads(Path(calibration_path).read_text()) if Path(calibration_path).exists() else None
    comparables_index = np.load(comparables_index_path, allow_pickle=True) if Path(comparables_index_path).exists() else None
    make_classifier = joblib.load(make_classifier_path) if Path(make_classifier_path).exists() else None
    class_classifier = joblib.load(class_classifier_path) if Path(class_classifier_path).exists() else None
    return {
        "device": device,
        "model": model,
        "preprocess": preprocess,
        "condition_text_pairs": condition_text_pairs,
        "bundle": bundle,
        "calibration": calibration,
        "comparables_index": comparables_index,
        "make_classifier": make_classifier,
        "class_classifier": class_classifier,
    }


def _predict_top_k(pooled_embedding: np.ndarray, spec_classifier: dict, label_key: str, top_k: int) -> list[dict]:
    clf, encoder = spec_classifier["classifier"], spec_classifier["label_encoder"]
    probs = clf.predict_proba(pooled_embedding.reshape(1, -1))[0]
    order = np.argsort(-probs)[:top_k]
    return [{label_key: encoder.classes_[i], "probability": float(probs[i])} for i in order]


def predict_make(pooled_embedding: np.ndarray, spec_classifier: dict, top_k: int = 2) -> list[dict]:
    """Trained (class-weighted logistic regression) make prediction, with
    calibrated probabilities -- unlike find_comparables()'s nearest-neighbor
    lookup, this can express "58% Ford, 35% Chevrolet" instead of a single
    falsely-confident guess when two makes look genuinely similar (e.g. Ford
    Econoline vs Chevrolet Express, both full-size cutaway vans -- a real
    reported miss that motivated adding this).

    Known limitation, also found via real testing: "make" itself can span
    visually unrelated body styles (a Ford Econoline van vs a Ford F-750
    conventional-cab truck look nothing alike), so this is confidently wrong
    for brands whose lineup is dominated by one body style in the training
    data. predict_gvwr_class() is the more robust fallback for exactly that
    case -- vehicle size/class is a consistent visual signal regardless of
    badge.
    """
    return _predict_top_k(pooled_embedding, spec_classifier, "make_name", top_k)


def predict_gvwr_class(pooled_embedding: np.ndarray, spec_classifier: dict, top_k: int = 2) -> list[dict]:
    """Trained GVWR weight-class prediction (e.g. "CLASS 6 (GVW 19501 - 26000)").
    70% accuracy on held-out data, with softer failure modes than make: it
    confuses adjacent weight classes, not unrelated vehicle shapes, because
    class correlates directly with visible vehicle size/proportions rather
    than a brand badge that can span multiple body styles.
    """
    return _predict_top_k(pooled_embedding, spec_classifier, "class_name", top_k)


def find_comparables(pooled_embedding: np.ndarray, comparables_index, k: int = 3) -> list[dict]:
    """Nearest neighbors (by CLIP embedding cosine similarity, both sides
    unit-normalized) among the train-split listings. This is NOT a trained
    classifier -- there isn't one for make/model/year in this pipeline --
    it's "which real listings in our data look most like this truck", used
    both to show checkable reasoning behind the price and, as a byproduct,
    a "predicted specs" readout: the closest match's year/make/model, for
    spot-checking whether the visual similarity is actually finding the
    right kind of truck.
    """
    embeddings = comparables_index["embeddings"]
    sims = embeddings @ pooled_embedding
    order = np.argsort(-sims)[:k]
    return [
        {
            "ad_id": str(comparables_index["ad_id"][i]),
            "year": int(comparables_index["year"][i]),
            "make_name": str(comparables_index["make_name"][i]),
            "model_name": str(comparables_index["model_name"][i]),
            "price": float(comparables_index["price"][i]),
            "similarity": float(sims[i]),
        }
        for i in order
    ]


def run_appraisal(paths: list[str], resources: dict) -> dict:
    """Runs the full pipeline on a listing's photos: gate -> embed -> condition
    -> price -> comparables -> predicted make/class. Returns {"gate",
    "condition", "price", "comparables", "predicted_make", "predicted_class"},
    with the latter five None when the gate rejects all photos.
    """
    device, model, preprocess = resources["device"], resources["model"], resources["preprocess"]
    gate_result = gate_images(paths, model, preprocess, device)

    if not gate_result["accepted"]:
        return {
            "gate": gate_result,
            "condition": None,
            "price": None,
            "comparables": None,
            "predicted_make": None,
            "predicted_class": None,
        }

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
    price_range = predict_range(bundle["model"], X, bundle["rel_lo"], bundle["rel_hi"])[0]
    price_result = adjust_price_range(price_range, gate_result)

    comparables_result = (
        find_comparables(pooled, resources["comparables_index"]) if resources["comparables_index"] is not None else None
    )
    predicted_make_result = (
        predict_make(pooled, resources["make_classifier"]) if resources["make_classifier"] is not None else None
    )
    predicted_class_result = (
        predict_gvwr_class(pooled, resources["class_classifier"]) if resources["class_classifier"] is not None else None
    )

    return {
        "gate": gate_result,
        "condition": condition_result,
        "price": price_result,
        "comparables": comparables_result,
        "predicted_make": predicted_make_result,
        "predicted_class": predicted_class_result,
    }
