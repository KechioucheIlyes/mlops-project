from typing import Any

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    status: str


class UploadResponse(BaseModel):
    candidate_id: str
    upload_dir: str
    run_name: str | None = None
    metric_name: str | None = None
    metric_value: float | None = None
    message: str


class PromoteRequest(BaseModel):
    candidate_id: str


class PromoteResponse(BaseModel):
    promoted: bool
    reason: str
    candidate_id: str
    production_metric_name: str | None = None
    production_metric_value: float | None = None
    candidate_metric_name: str | None = None
    candidate_metric_value: float | None = None
    production_dir: str
    archive_dir: str | None = None
    current_model: dict[str, Any] | None = None

    model_config = ConfigDict(extra="ignore")