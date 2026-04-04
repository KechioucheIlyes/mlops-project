from prometheus_client import Counter, Histogram

PREDICT_REQUESTS_TOTAL = Counter(
    "predict_requests_total",
    "Total number of predict requests",
    ["status"],
)

PREDICT_REQUEST_DURATION_SECONDS = Histogram(
    "predict_request_duration_seconds",
    "Duration of predict requests",
)