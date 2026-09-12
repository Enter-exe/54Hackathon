"""Load appraisal artifacts once and compose a UI-ready truck appraisal."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import joblib
import torch

from modeling.comparables import find_comparables, load_comparables
from modeling.confidence import adjust_price_range
from modeling.train_price import predict_quantiles
from pipeline.build_artifacts import EXPECTED
from pipeline.condition_assessment import assess_images, load_text_embeddings
from pipeline.extract_embeddings import MODEL_NAME, load_model, pooled_listing_embedding
from pipeline.gate_images import gate_images
from pipeline.gating import build_gate


REQUIRED_ARTIFACTS = tuple(EXPECTED.values())


def artifacts_ready(repo_root: Path) -> bool:
    """Return whether every output of the artifact build exists as a file."""
    root = Path(repo_root)
    return all((root / relative).is_file() for relative in REQUIRED_ARTIFACTS)


@dataclass
class AppraisalEngine:
    model: object
    preprocess: object
    device: str
    price_models: object
    condition_text_pairs: object
    condition_thresholds: object
    photo_gate: object
    comparable_index: dict
    listings_by_id: dict

    @classmethod
    def from_artifacts(cls, repo_root: Path) -> Self:
        """Load static model resources and generated artifacts for reuse."""
        root = Path(repo_root)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = load_model(device)
        price_bundle = joblib.load(root / EXPECTED["price_models"])
        calibration = json.loads(
            (root / EXPECTED["condition_calibration"]).read_text()
        )
        comparable_index, listings_by_id = load_comparables(
            root / EXPECTED["comparables"], root / EXPECTED["listings"]
        )
        return cls(
            model=model,
            preprocess=preprocess,
            device=device,
            price_models=price_bundle["models"],
            condition_text_pairs=load_text_embeddings(model, device),
            condition_thresholds=calibration["thresholds"],
            photo_gate=build_gate(model, device, MODEL_NAME),
            comparable_index=comparable_index,
            listings_by_id=listings_by_id,
        )

    def appraise(self, image_paths: list[str]) -> dict:
        """Return a complete accepted or rejected consumer-facing appraisal."""
        gate_result = gate_images(
            image_paths, self.model, self.preprocess, self.device
        )
        if not gate_result["accepted"]:
            return self._rejected(gate_result)

        embedding = pooled_listing_embedding(
            gate_result["usable_paths"],
            self.model,
            self.preprocess,
            self.device,
            gate=self.photo_gate,
        )
        if embedding is None:
            return self._rejected(
                {
                    **gate_result,
                    "accepted": False,
                    "confidence": "low",
                    "reasons": ["no_usable_truck_photos"],
                    "rejected": [
                        *gate_result.get("rejected", []),
                        *[
                            {
                                "path": path,
                                "reasons": ["no_usable_truck_photos"],
                            }
                            for path in gate_result["usable_paths"]
                        ],
                    ],
                }
            )

        prediction = predict_quantiles(
            self.price_models, embedding[None, :]
        )[0]
        condition = assess_images(
            gate_result["usable_paths"],
            self.model,
            self.preprocess,
            self.device,
            self.condition_text_pairs,
            thresholds=self.condition_thresholds,
            gate=self.photo_gate,
        )
        return {
            **adjust_price_range(prediction, gate_result),
            "condition": condition or {},
            "comparables": find_comparables(
                embedding, self.comparable_index, self.listings_by_id, k=3
            ),
        }

    @staticmethod
    def _rejected(gate_result: dict) -> dict:
        return {
            **adjust_price_range([], gate_result),
            "rejected": gate_result.get("rejected", []),
            "condition": {},
            "comparables": [],
        }
