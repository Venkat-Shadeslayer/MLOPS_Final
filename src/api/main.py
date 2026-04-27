"""
FastAPI inference service.

Endpoints:
  GET  /health        liveness probe (cheap)
  GET  /ready         readiness probe (model loaded?)
  POST /predict       single-reading AQI prediction
  POST /ground-truth  report actual AQI for a past prediction
  GET  /stats         rolling RMSE + recent counts
  GET  /metrics       Prometheus exposition (auto)
  GET  /docs          OpenAPI Swagger UI (auto)

Design notes:
  - Loose coupling: frontend never touches MLflow or Postgres directly.
  - Every prediction is logged to Postgres; this drives the feedback loop.
  - Prometheus instrumentator handles HTTP metrics; custom gauges/counters
    in instrumentation.py cover ML-specific signals.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.instrumentation import (
    GROUND_TRUTH_SUBMISSIONS,
    PREDICTION_LATENCY,
    PREDICTION_VALUE,
    PREDICTIONS_TOTAL,
    ROLLING_RMSE,
)
from src.api.model_loader import model_loader
from src.api.predictions_db import (
    feedback_count,
    init_db,
    insert_prediction,
    list_feedback,
    record_ground_truth,
    rolling_rmse,
)
from src.utils.config import drift_config
from src.api.schemas import (
    GroundTruthSubmission,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    ReadyResponse,
)
from src.utils.logging import get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hooks."""
    log.info("Starting up")
    init_db()
    model_loader.load()  # best-effort; /ready reports status
    yield
    log.info("Shutting down")


app = FastAPI(
    title="AQI Prediction API",
    description="AQI regression inference service.",
    version="0.1.0",
    lifespan=lifespan,
)

# Auto-instrument HTTP metrics. Exposes /metrics.
# Add service="inference" label to match prometheus.yml scrape config.
instrumentator = Instrumentator(
    should_group_status_codes=False,
    excluded_handlers=["/metrics", "/health", "/ready", "/docs", "/openapi.json"],
)
instrumentator.instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


# --------------------------------------------------------------------------
# Feature construction from a PredictionRequest
# --------------------------------------------------------------------------
POLLUTANT_ALIASES = {
    "pm25": "PM2.5", "pm10": "PM10", "no": "NO", "no2": "NO2", "nox": "NOx",
    "nh3": "NH3", "co": "CO", "so2": "SO2", "o3": "O3",
    "benzene": "Benzene", "toluene": "Toluene", "xylene": "Xylene",
}


def build_feature_row(req: PredictionRequest, feature_cols: list[str]) -> pd.DataFrame:
    """
    Assemble a single-row DataFrame with columns matching the model's
    training signature. Missing lag/rolling values are zero-filled; in a
    production system these would come from a feature store.
    """
    current = req.reading.model_dump(by_alias=True)
    row = dict.fromkeys(feature_cols, 0.0)

    # Fill current pollutant values
    for _alias_key, canonical in POLLUTANT_ALIASES.items():
        val = current.get(canonical)
        if val is not None and canonical in row:
            row[canonical] = float(val)

    # Compute lags from history if provided
    if req.history:
        hist_by_date = sorted(req.history, key=lambda r: r.date, reverse=True)
        for lag_idx, past in enumerate(hist_by_date[:14], start=1):
            past_dict = past.model_dump(by_alias=True)
            for canonical in POLLUTANT_ALIASES.values():
                lag_col = f"{canonical}_lag{lag_idx}"
                if lag_col in row and past_dict.get(canonical) is not None:
                    row[lag_col] = float(past_dict[canonical])

    return pd.DataFrame([row])[feature_cols].astype("float32")


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health():
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse, tags=["health"])
async def ready():
    if model_loader.is_loaded:
        return ReadyResponse(
            ready=True,
            model_loaded=True,
            model_name="aqi_regressor",
            model_version=model_loader.version,
        )
    return ReadyResponse(
        ready=False,
        model_loaded=False,
        detail=model_loader.load_error or "Model not loaded yet",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(req: PredictionRequest):
    if not model_loader.is_loaded:
        # Attempt a reload (model may have been registered after startup)
        model_loader.load()
        if not model_loader.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    try:
        features_df = build_feature_row(req, model_loader.feature_cols)
        pred = model_loader.predict(features_df)
    except Exception as e:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}") from e

    latency_s = time.perf_counter() - start
    latency_ms = latency_s * 1000

    # Persist + metrics
    pred_id = insert_prediction(
        model_version=model_loader.version or "unknown",
        model_family=model_loader.family or "unknown",
        input_features=req.reading.model_dump(by_alias=True),
        predicted_aqi=pred,
        latency_ms=latency_ms,
    )
    PREDICTIONS_TOTAL.labels(
        model_version=model_loader.version or "unknown",
        model_family=model_loader.family or "unknown",
    ).inc()
    PREDICTION_LATENCY.observe(latency_s)
    PREDICTION_VALUE.observe(pred)

    import datetime as _dt
    return PredictionResponse(
        prediction_id=pred_id,
        predicted_aqi=pred,
        model_version=model_loader.version or "unknown",
        model_stage="Production",
        timestamp=_dt.datetime.utcnow(),
        latency_ms=round(latency_ms, 3),
    )


@app.post("/ground-truth", tags=["feedback"])
async def submit_ground_truth(payload: GroundTruthSubmission):
    found = record_ground_truth(payload.prediction_id, payload.actual_aqi)
    if not found:
        raise HTTPException(status_code=404, detail="prediction_id not found")
    GROUND_TRUTH_SUBMISSIONS.inc()
    # Refresh the rolling RMSE gauge
    rmse = rolling_rmse(window_hours=24)
    if rmse is not None:
        ROLLING_RMSE.set(rmse)
    return {"status": "recorded", "rolling_rmse_24h": rmse}


@app.get("/stats", tags=["feedback"])
async def stats():
    """Rolling RMSE + feedback count + retrain-trigger config.

    Feedback-based retrain fires when feedback_count >= feedback_count_threshold
    AND rolling_rmse > rmse_threshold (within the window).
    """
    window = drift_config.check_window_hours
    rmse = rolling_rmse(window_hours=window)
    count = feedback_count(window_hours=window)
    return {
        "rolling_rmse_24h": rmse,
        "feedback_count_window": count,
        "window_hours": window,
        "feedback_count_threshold": drift_config.feedback_count_threshold,
        "rmse_threshold": drift_config.rmse_threshold,
        "psi_threshold": drift_config.psi_threshold,
        "count_gate_met": count >= drift_config.feedback_count_threshold,
        "rmse_gate_met": rmse is not None and rmse > drift_config.rmse_threshold,
    }


@app.get("/feedback", tags=["feedback"])
async def feedback_list(limit: int = 500):
    """Return recent feedback rows (predictions with actual_aqi)."""
    rows = list_feedback(limit=limit)
    # Stringify non-JSON types for the wire
    out = []
    for r in rows:
        out.append({
            "prediction_id": str(r["id"]),
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "feedback_at": r["feedback_at"].isoformat() if r["feedback_at"] else None,
            "model_version": r["model_version"],
            "model_family": r["model_family"],
            "predicted_aqi": r["predicted_aqi"],
            "actual_aqi": r["actual_aqi"],
            "error": r["actual_aqi"] - r["predicted_aqi"],
            "abs_error": abs(r["actual_aqi"] - r["predicted_aqi"]),
            "latency_ms": r["latency_ms"],
            "city": (r["input_features"] or {}).get("city"),
            "date": (r["input_features"] or {}).get("date"),
        })
    return {"count": len(out), "rows": out}


@app.get("/feedback.csv", tags=["feedback"])
async def feedback_csv(limit: int = 10000):
    """Download all feedback rows as CSV (for audit / external analysis)."""
    import csv
    import io

    from fastapi.responses import StreamingResponse

    rows = list_feedback(limit=limit)
    buf = io.StringIO()
    fieldnames = [
        "prediction_id", "created_at", "feedback_at",
        "model_version", "model_family",
        "city", "date",
        "predicted_aqi", "actual_aqi", "error", "abs_error",
        "latency_ms",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        feat = r["input_features"] or {}
        writer.writerow({
            "prediction_id": str(r["id"]),
            "created_at": r["created_at"].isoformat() if r["created_at"] else "",
            "feedback_at": r["feedback_at"].isoformat() if r["feedback_at"] else "",
            "model_version": r["model_version"],
            "model_family": r["model_family"],
            "city": feat.get("city", ""),
            "date": feat.get("date", ""),
            "predicted_aqi": r["predicted_aqi"],
            "actual_aqi": r["actual_aqi"],
            "error": r["actual_aqi"] - r["predicted_aqi"],
            "abs_error": abs(r["actual_aqi"] - r["predicted_aqi"]),
            "latency_ms": r["latency_ms"],
        })
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=feedback.csv"},
    )
