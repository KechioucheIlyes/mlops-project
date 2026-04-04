from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    candidate_id: str | None
    run_name: str | None
    metric_name: str | None
    metric_value: float | None
    model_path: str
    results_path: str | None
    classes: list[str]


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    model: dict