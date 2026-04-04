from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config import get_settings
from app.inference import (
    CLASSES,
    get_model_metadata,
    get_model_paths,
    get_results_metadata,
    load_model_once,
    predict_tensor,
)
from app.metrics import PREDICT_REQUEST_DURATION_SECONDS, PREDICT_REQUESTS_TOTAL
from app.preprocessing import load_image_as_tensor
from app.schemas import HealthResponse, ModelInfoResponse, PredictResponse

app = FastAPI(title="Predict API", version="1.0.0")


@app.on_event("startup")
def startup() -> None:
    load_model_once()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    model_path, _metadata_path, results_path = get_model_paths()
    metadata = get_model_metadata()
    results = get_results_metadata()

    return ModelInfoResponse(
        candidate_id=metadata.get("candidate_id"),
        run_name=metadata.get("run_name"),
        metric_name=metadata.get("metric_name"),
        metric_value=metadata.get("metric_value"),
        model_path=str(model_path),
        results_path=str(results_path) if results else None,
        classes=results.get("classes", CLASSES),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(file: UploadFile = File(...)) -> PredictResponse:
    with PREDICT_REQUEST_DURATION_SECONDS.time():
        try:
            image_bytes = file.file.read()
            if not image_bytes:
                raise HTTPException(status_code=400, detail="Empty file.")

            settings = get_settings()
            image_tensor = load_image_as_tensor(
                image_bytes=image_bytes,
                image_size=settings.image_size,
            )

            prediction = predict_tensor(image_tensor)
            metadata = get_model_metadata()

            PREDICT_REQUESTS_TOTAL.labels(status="success").inc()

            return PredictResponse(
                predicted_class=prediction["predicted_class"],
                confidence=prediction["confidence"],
                probabilities=prediction["probabilities"],
                model={
                    "candidate_id": metadata.get("candidate_id"),
                    "run_name": metadata.get("run_name"),
                    "metric_name": metadata.get("metric_name"),
                    "metric_value": metadata.get("metric_value"),
                },
            )
        except HTTPException:
            PREDICT_REQUESTS_TOTAL.labels(status="error").inc()
            raise
        except Exception as e:
            PREDICT_REQUESTS_TOTAL.labels(status="error").inc()
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)