"""
Drift detection via Population Stability Index (PSI).

PSI compares two distributions (baseline vs current) using the same bin edges.
Convention:
  PSI < 0.10  : no significant drift
  0.10 - 0.25 : moderate drift, investigate
  PSI > 0.25  : significant drift, likely needs retraining
Our threshold default is 0.20 (middle of moderate), overridable via params.yaml.

Baseline was computed during Day 2 feature engineering and saved to
data/processed/baseline_stats.json. We compare recent prediction inputs
(last N hours from the predictions table) against those reference
distributions.

PSI values get pushed to Prometheus pushgateway so Grafana + alert rules
can consume them.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from sqlalchemy import text

from src.api.predictions_db import get_session, init_db
from src.utils.config import BASELINE_STATS_PATH, POLLUTANT_FEATURES
from src.utils.logging import get_logger

log = get_logger(__name__)

PUSHGATEWAY_URL = "pushgateway:9091"
JOB_NAME = "aqi_drift_monitor"


def compute_psi(
    current: np.ndarray,
    bin_edges: list[float],
    reference_dist: list[float],
    epsilon: float = 1e-4,
) -> float:
    """
    Population Stability Index.

    PSI = sum over bins of (current_pct - ref_pct) * ln(current_pct / ref_pct)
    Epsilon prevents divide-by-zero when a bin is empty in one distribution.
    """
    if len(current) == 0:
        return 0.0
    counts, _ = np.histogram(current, bins=bin_edges)
    total = counts.sum()
    if total == 0:
        return 0.0
    current_dist = counts / total

    ref = np.array(reference_dist)
    cur = np.array(current_dist)
    # Pad if bin counts differ (shouldn't happen if we use the same edges)
    n = min(len(ref), len(cur))
    ref = ref[:n]
    cur = cur[:n]

    # Replace zeros to avoid log(0) and division-by-zero
    ref = np.where(ref == 0, epsilon, ref)
    cur = np.where(cur == 0, epsilon, cur)

    return float(np.sum((cur - ref) * np.log(cur / ref)))


def load_baseline() -> dict:
    if not BASELINE_STATS_PATH.exists():
        raise FileNotFoundError(
            f"No baseline stats at {BASELINE_STATS_PATH}. Run the data pipeline first."
        )
    with BASELINE_STATS_PATH.open() as f:
        return json.load(f)["feature_stats"]


def fetch_recent_inputs(hours: int = 24) -> pd.DataFrame:
    """Pull recent prediction inputs from the predictions table."""
    init_db()
    sql = text(
        """
        SELECT input_features
        FROM predictions
        WHERE created_at > NOW() - (:hours || ' hours')::interval
        """
    )
    with get_session() as s:
        rows = s.execute(sql, {"hours": hours}).fetchall()
    if not rows:
        return pd.DataFrame()
    records = [r[0] for r in rows]  # input_features is JSONB
    df = pd.DataFrame.from_records(records)
    # Coerce numeric columns — JSONB stores numbers as strings sometimes
    for col in POLLUTANT_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compute_drift_report(hours: int = 24) -> dict:
    """Compute PSI per pollutant feature. Returns {feature: psi}."""
    baseline = load_baseline()
    recent = fetch_recent_inputs(hours)
    if recent.empty:
        log.warning("No recent predictions to compute drift against.")
        return {}

    report: dict = {}
    for feature in POLLUTANT_FEATURES:
        if feature not in recent.columns or feature not in baseline:
            continue
        current = recent[feature].dropna().to_numpy()
        stats = baseline[feature]
        psi = compute_psi(
            current,
            bin_edges=stats["bin_edges"],
            reference_dist=stats["reference_distribution"],
        )
        report[feature] = psi
    return report


def push_drift_metrics(report: dict) -> None:
    """Push PSI gauges to pushgateway for Prometheus scraping."""
    registry = CollectorRegistry()
    psi_gauge = Gauge(
        "aqi_feature_psi",
        "Population Stability Index per feature (vs training baseline)",
        labelnames=["feature"],
        registry=registry,
    )
    max_psi_gauge = Gauge(
        "aqi_feature_psi_max",
        "Max PSI across all features",
        registry=registry,
    )
    for feature, psi in report.items():
        psi_gauge.labels(feature=feature).set(psi)
    if report:
        max_psi_gauge.set(max(report.values()))
    push_to_gateway(PUSHGATEWAY_URL, job=JOB_NAME, registry=registry)
    log.info("Pushed PSI metrics for %d features (max=%.3f)",
             len(report), max(report.values()) if report else 0.0)


def main() -> dict:
    """Airflow entrypoint. Computes drift + pushes to Prometheus."""
    report = compute_drift_report(hours=24)
    if not report:
        log.info("No drift report generated (no recent predictions).")
        return {}
    for feat, psi in sorted(report.items(), key=lambda kv: -kv[1]):
        log.info("PSI[%s] = %.4f", feat, psi)
    push_drift_metrics(report)
    return report


if __name__ == "__main__":
    main()
