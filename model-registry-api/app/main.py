from __future__ import annotations

import uuid

from fastapi import FastAPI, File, Form, Header, UploadFile
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.metrics import (
    ARCHIVED_MODELS_TOTAL,
    PROMOTED_MODELS_TOTAL,
    PROMOTION_REQUESTS_TOTAL,
    REQUEST_DURATION_SECONDS,
    UPLOAD_REQUESTS_TOTAL,
)
from app.models import HealthResponse, PromoteRequest, PromoteResponse, UploadResponse
from app.services import (
    get_archive_root,
    get_production_root,
    get_uploads_root,
    promote_candidate,
    save_candidate_files,
    validate_token,
)

app = FastAPI(title="Model Registry API", version="1.0.0")


@app.on_event("startup")
def startup() -> None:
    get_uploads_root()
    get_production_root()
    get_archive_root()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/upload-model", response_model=UploadResponse)
def upload_model(
    authorization: str | None = Header(default=None),
    run_name: str | None = Form(default=None),
    candidate_id: str | None = Form(default=None),
    results_file: UploadFile = File(...),
    best_model_file: UploadFile = File(...),
) -> UploadResponse:
    with REQUEST_DURATION_SECONDS.labels("/upload-model").time():
        try:
            validate_token(authorization)
            effective_candidate_id = candidate_id or uuid.uuid4().hex
            dest_dir, _results, metric_name, metric_value = save_candidate_files(
                candidate_id=effective_candidate_id,
                run_name=run_name,
                results_file=results_file,
                best_model_file=best_model_file,
            )
            UPLOAD_REQUESTS_TOTAL.labels(status="success").inc()
            return UploadResponse(
                candidate_id=effective_candidate_id,
                upload_dir=str(dest_dir),
                run_name=run_name,
                metric_name=metric_name,
                metric_value=metric_value,
                message="Model candidate uploaded successfully.",
            )
        except Exception:
            UPLOAD_REQUESTS_TOTAL.labels(status="error").inc()
            raise


@app.post("/promote-model", response_model=PromoteResponse)
def promote_model(
    payload: PromoteRequest,
    authorization: str | None = Header(default=None),
) -> PromoteResponse:
    with REQUEST_DURATION_SECONDS.labels("/promote-model").time():
        try:
            validate_token(authorization)
            result = promote_candidate(payload.candidate_id)

            if result["promoted"]:
                PROMOTION_REQUESTS_TOTAL.labels(status="promoted").inc()
                PROMOTED_MODELS_TOTAL.inc()
                if result["archive_dir"]:
                    ARCHIVED_MODELS_TOTAL.inc()
            else:
                PROMOTION_REQUESTS_TOTAL.labels(status="kept_current").inc()

            return PromoteResponse(
                promoted=result["promoted"],
                reason=result["reason"],
                candidate_id=payload.candidate_id,
                production_metric_name=result["production_metric_name"],
                production_metric_value=result["production_metric_value"],
                candidate_metric_name=result["candidate_metric_name"],
                candidate_metric_value=result["candidate_metric_value"],
                production_dir="/registry/production",
                archive_dir=result["archive_dir"],
                current_model=result["current_model"],
            )
        except Exception:
            PROMOTION_REQUESTS_TOTAL.labels(status="error").inc()
            raise