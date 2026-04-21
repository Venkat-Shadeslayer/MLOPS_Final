"""
Performance decay check.

Two signals trigger retraining:
  1. Rolling RMSE > threshold (model accuracy degraded per ground truth)
  2. Max feature PSI > threshold (input distributions have shifted)

Returns a dict with both values and a boolean 'should_retrain'. The Airflow
DAG uses this to decide whether to fire the training_pipeline.
"""
from __future__ import annotations

from src.api.predictions_db import rolling_rmse
from src.monitoring.drift import compute_drift_report
from src.utils.config import drift_config
from src.utils.logging import get_logger

log = get_logger(__name__)


def check_decay() -> dict:
    """Compute decay indicators and return a retrain decision."""
    rmse = rolling_rmse(window_hours=drift_config.check_window_hours)
    drift_report = compute_drift_report(hours=drift_config.check_window_hours)
    max_psi = max(drift_report.values()) if drift_report else 0.0

    rmse_breach = rmse is not None and rmse > drift_config.rmse_threshold
    psi_breach = max_psi > drift_config.psi_threshold

    decision = {
        "rolling_rmse": rmse,
        "rmse_threshold": drift_config.rmse_threshold,
        "max_psi": max_psi,
        "psi_threshold": drift_config.psi_threshold,
        "rmse_breach": rmse_breach,
        "psi_breach": psi_breach,
        "should_retrain": rmse_breach or psi_breach,
    }
    log.info("Decay check: %s", decision)
    return decision


if __name__ == "__main__":
    result = check_decay()
    print(result)
