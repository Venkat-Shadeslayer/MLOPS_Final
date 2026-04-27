"""
Performance decay check.

Retrain triggers (OR):
  1. Feedback-RMSE gate: at least N feedback rows in the window AND
     rolling RMSE over those rows > threshold.
     Both N (FEEDBACK_COUNT_THRESHOLD) and threshold (DRIFT_RMSE_THRESHOLD)
     are configurable via .env. The count gate avoids a single outlier
     feedback triggering an expensive retrain.
  2. Input-drift gate: max per-feature PSI > threshold (data distribution
     shift — fires even without any feedback).

Returns a dict that the drift_monitor DAG branches on.
"""
from __future__ import annotations

from src.api.predictions_db import feedback_count, rolling_rmse
from src.monitoring.drift import compute_drift_report
from src.utils.config import drift_config
from src.utils.logging import get_logger

log = get_logger(__name__)


def check_decay() -> dict:
    """Compute decay indicators and return a retrain decision."""
    rmse = rolling_rmse(window_hours=drift_config.check_window_hours)
    fb_count = feedback_count(window_hours=drift_config.check_window_hours)
    drift_report = compute_drift_report(hours=drift_config.check_window_hours)
    max_psi = max(drift_report.values()) if drift_report else 0.0

    count_gate_met = fb_count >= drift_config.feedback_count_threshold
    rmse_gate_met = rmse is not None and rmse > drift_config.rmse_threshold
    feedback_breach = count_gate_met and rmse_gate_met
    psi_breach = max_psi > drift_config.psi_threshold

    decision = {
        "feedback_count": fb_count,
        "feedback_count_threshold": drift_config.feedback_count_threshold,
        "rolling_rmse": rmse,
        "rmse_threshold": drift_config.rmse_threshold,
        "max_psi": max_psi,
        "psi_threshold": drift_config.psi_threshold,
        "window_hours": drift_config.check_window_hours,
        "count_gate_met": count_gate_met,
        "rmse_gate_met": rmse_gate_met,
        "feedback_breach": feedback_breach,
        "psi_breach": psi_breach,
        "should_retrain": feedback_breach or psi_breach,
    }
    log.info("Decay check: %s", decision)
    return decision


if __name__ == "__main__":
    result = check_decay()
    print(result)
