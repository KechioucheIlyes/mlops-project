from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    registry_production_dir: Path
    runtime_production_dir: Path
    model_filename: str
    metadata_filename: str
    results_filename: str
    image_size: int
    device: str


def get_settings() -> Settings:
    return Settings(
        registry_production_dir=Path(
            os.getenv("REGISTRY_PRODUCTION_DIR", "/app/storage/registry/production")
        ),
        runtime_production_dir=Path(
            os.getenv("RUNTIME_PRODUCTION_DIR", "/app/runtime-model/production")
        ),
        model_filename=os.getenv("MODEL_FILENAME", "best_model.pth"),
        metadata_filename=os.getenv("MODEL_METADATA_FILENAME", "current_model.json"),
        results_filename=os.getenv("MODEL_RESULTS_FILENAME", "results.json"),
        image_size=int(os.getenv("IMG_SIZE", "224")),
        device=os.getenv("PREDICT_DEVICE", "cpu"),
    )