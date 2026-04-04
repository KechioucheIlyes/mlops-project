from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from shifaa.vision import VisionModelFactory

from app.config import get_settings

CLASSES = ["COVID", "Lung_Opacity", "Normal"]

_MODEL = None
_MODEL_META = None
_RESULTS = None


def load_shifaa_backbone(num_classes: int = 3) -> torch.nn.Module:
    shifaa_model = VisionModelFactory.create_model(
        model_type="classification",
        model_name="Chest_COVID",
    )

    base_model = shifaa_model.model.model
    in_features = base_model.fc.in_features
    base_model.fc = nn.Linear(in_features, num_classes)
    return base_model


def get_model_paths() -> tuple[Path, Path, Path]:
    settings = get_settings()
    production_dir = settings.registry_production_dir

    model_path = production_dir / settings.model_filename
    metadata_path = production_dir / settings.metadata_filename
    results_path = production_dir / settings.results_filename

    return model_path, metadata_path, results_path


def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_model_once():
    global _MODEL, _MODEL_META, _RESULTS

    if _MODEL is not None:
        return _MODEL

    settings = get_settings()
    model_path, metadata_path, results_path = get_model_paths()

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = torch.device(settings.device)

    model = load_shifaa_backbone(num_classes=3)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    _MODEL = model
    _MODEL_META = load_json_if_exists(metadata_path)
    _RESULTS = load_json_if_exists(results_path)

    return _MODEL


def get_loaded_model():
    model = load_model_once()
    return model


def get_model_metadata() -> dict:
    load_model_once()
    return _MODEL_META or {}


def get_results_metadata() -> dict:
    load_model_once()
    return _RESULTS or {}


@torch.no_grad()
def predict_tensor(image_tensor: torch.Tensor) -> dict:
    settings = get_settings()
    device = torch.device(settings.device)

    model = get_loaded_model()
    image_tensor = image_tensor.to(device)

    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1)[0]
    confidence, predicted_idx = torch.max(probabilities, dim=0)

    probs = {
        CLASSES[i]: float(probabilities[i].item())
        for i in range(len(CLASSES))
    }

    return {
        "predicted_class": CLASSES[int(predicted_idx.item())],
        "confidence": float(confidence.item()),
        "probabilities": probs,
    }