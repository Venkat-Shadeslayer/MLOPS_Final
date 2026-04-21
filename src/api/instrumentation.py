"""
Custom Prometheus metrics for the API.

The FastAPI instrumentator handles http_* metrics automatically.
Here we add ML-specific metrics that the Grafana dashboard and
Prometheus alert rules reference:

  - aqi_predictions_total{model_version, model_family}
  - aqi_prediction_latency_seconds (histogram)
  - aqi_prediction_value (histogram of predicted AQIs — detects distribution drift)
  - aqi_rolling_rmse (gauge, refreshed periodically from DB)
  - aqi_feature_psi{feature} (gauge, populated by drift detection in Day 5)
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

PREDICTIONS_TOTAL = Counter(
    "aqi_predictions_total",
    "Total number of AQI predictions served",
    ["model_version", "model_family"],
)

PREDICTION_LATENCY = Histogram(
    "aqi_prediction_latency_seconds",
    "Latency of AQI prediction calls",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

PREDICTION_VALUE = Histogram(
    "aqi_prediction_value",
    "Distribution of predicted AQI values",
    buckets=(0, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000),
)

ROLLING_RMSE = Gauge(
    "aqi_rolling_rmse",
    "Rolling RMSE between predicted and actual AQI over the feedback window",
)

FEATURE_PSI = Gauge(
    "aqi_feature_psi",
    "Population Stability Index per feature (vs training baseline)",
    ["feature"],
)

GROUND_TRUTH_SUBMISSIONS = Counter(
    "aqi_ground_truth_submissions_total",
    "Count of ground-truth reports received",
)
