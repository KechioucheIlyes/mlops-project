from prometheus_client import Counter, Histogram

UPLOAD_REQUESTS_TOTAL = Counter(
    "model_registry_upload_requests_total",
    "Total number of upload-model requests",
    ["status"],
)

PROMOTION_REQUESTS_TOTAL = Counter(
    "model_registry_promote_requests_total",
    "Total number of promote-model requests",
    ["status"],
)

PROMOTED_MODELS_TOTAL = Counter(
    "model_registry_promoted_models_total",
    "Total number of models promoted to production",
)

ARCHIVED_MODELS_TOTAL = Counter(
    "model_registry_archived_models_total",
    "Total number of production models archived",
)

REQUEST_DURATION_SECONDS = Histogram(
    "model_registry_request_duration_seconds",
    "Request duration in seconds",
    ["endpoint"],
)