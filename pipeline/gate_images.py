"""Reject low-quality or irrelevant images before truck appraisal."""

import numpy as np
import open_clip
import torch
from PIL import Image

from pipeline.extract_embeddings import MODEL_NAME, embed_images


DARK_THRESHOLD = 35.0
BLUR_THRESHOLD = 40.0
TRUCK_MARGIN = 0.02
TRUCK_PROMPTS = (
    "a photo of a commercial truck",
    "a photo of a box truck",
    "a photo of a semi tractor",
)
NON_TRUCK_PROMPTS = (
    "a photo of a motorcycle",
    "a photo of a passenger car",
    "a photo of a person",
    "a photo of an indoor room",
    "an unclear or irrelevant photo",
)


def assess_image_quality(
    path,
    dark_threshold=DARK_THRESHOLD,
    blur_threshold=BLUR_THRESHOLD,
):
    """Measure brightness and Laplacian sharpness for one image."""
    try:
        with Image.open(path) as image:
            gray = np.asarray(image.convert("L"), dtype=np.float32)
    except Exception:
        return {
            "path": str(path),
            "brightness": None,
            "sharpness": None,
            "reasons": ["unreadable"],
        }

    brightness = float(gray.mean())
    if min(gray.shape) < 3:
        sharpness = 0.0
    else:
        center = gray[1:-1, 1:-1]
        laplacian = (
            4 * center
            - gray[:-2, 1:-1]
            - gray[2:, 1:-1]
            - gray[1:-1, :-2]
            - gray[1:-1, 2:]
        )
        sharpness = float(laplacian.var())

    reasons = []
    if brightness < dark_threshold:
        reasons.append("too_dark")
    if sharpness < blur_threshold:
        reasons.append("too_blurry")
    return {
        "path": str(path),
        "brightness": brightness,
        "sharpness": sharpness,
        "reasons": reasons,
    }


def similarity_margins(image_features, text_features, positive_count):
    """Compare each image's best truck and non-truck prompt scores."""
    if not 0 < positive_count < len(text_features):
        raise ValueError("positive_count must leave positive and negative prompts")
    similarities = image_features @ text_features.T
    return (
        similarities[:, :positive_count].max(axis=1)
        - similarities[:, positive_count:].max(axis=1)
    )


@torch.no_grad()
def truck_similarity_margins(paths, model, preprocess, device):
    """Return CLIP truck-vs-non-truck similarity margins for image paths."""
    image_features = embed_images(paths, model, preprocess, device)
    prompts = TRUCK_PROMPTS + NON_TRUCK_PROMPTS
    tokens = open_clip.get_tokenizer(MODEL_NAME)(prompts).to(device)
    text_features = model.encode_text(tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return similarity_margins(
        image_features,
        text_features.cpu().numpy(),
        len(TRUCK_PROMPTS),
    )


def gate_images(
    paths,
    model,
    preprocess,
    device,
    dark_threshold=DARK_THRESHOLD,
    blur_threshold=BLUR_THRESHOLD,
    truck_margin=TRUCK_MARGIN,
):
    """Return usable truck photos and reasons for every rejected input."""
    paths = [str(path) for path in paths]
    assessments = [
        assess_image_quality(path, dark_threshold, blur_threshold) for path in paths
    ]
    candidates = [item["path"] for item in assessments if not item["reasons"]]
    rejected = [item for item in assessments if item["reasons"]]

    usable_paths = []
    if candidates:
        margins = truck_similarity_margins(candidates, model, preprocess, device)
        if len(margins) != len(candidates):
            raise ValueError("CLIP returned the wrong number of image scores")
        if np.max(margins) >= truck_margin:
            usable_paths = candidates
        else:
            for path, margin in zip(candidates, margins):
                rejected.append(
                    {
                        "path": path,
                        "brightness": None,
                        "sharpness": None,
                        "truck_margin": float(margin),
                        "reasons": ["not_truck"],
                    }
                )

    accepted = bool(usable_paths)
    if len(usable_paths) < 3:
        confidence = "low"
    elif rejected:
        confidence = "medium"
    else:
        confidence = "high"

    warnings = []
    if rejected:
        warnings.append("some_photos_rejected")
    if accepted and len(usable_paths) < 3:
        warnings.append("limited_photo_coverage")
    return {
        "accepted": accepted,
        "confidence": confidence,
        "usable_paths": usable_paths,
        "rejected": rejected,
        "warnings": warnings,
        "reasons": [] if accepted else ["no_usable_truck_photos"],
    }
