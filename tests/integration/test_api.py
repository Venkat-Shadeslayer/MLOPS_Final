"""
Integration tests against a running API container.

Run: docker compose up -d && pytest tests/integration -v

Skipped automatically if the API is unreachable — keeps CI green when
nobody has brought the stack up.
"""
from __future__ import annotations

import os
import uuid

import pytest
import requests

API = os.getenv("API_URL", "http://localhost:8000")


def _api_up() -> bool:
    try:
        return requests.get(f"{API}/health", timeout=2).status_code == 200
    except requests.RequestException:
        return False


pytestmark = pytest.mark.skipif(not _api_up(), reason="API not reachable")


def test_health():
    r = requests.get(f"{API}/health", timeout=5)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready_shape():
    r = requests.get(f"{API}/ready", timeout=5)
    assert r.status_code == 200
    body = r.json()
    assert "ready" in body
    assert "model_loaded" in body


def _sample_reading() -> dict:
    return {
        "reading": {
            "city": "Delhi", "date": "2025-04-23",
            "PM2.5": 110.5, "PM10": 180.0, "NO": 15.2, "NO2": 40.1,
            "NOx": 55.0, "NH3": 12.3, "CO": 1.1, "SO2": 8.2,
            "O3": 25.0, "Benzene": 2.1, "Toluene": 5.3, "Xylene": 1.4,
        },
        "history": [],
    }


def test_predict_valid_payload():
    r = requests.post(f"{API}/predict", json=_sample_reading(), timeout=10)
    if r.status_code == 503:
        pytest.skip("Model not loaded — run training_pipeline first")
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["predicted_aqi"] <= 1000.0
    uuid.UUID(body["prediction_id"])  # raises if not a UUID


def test_predict_with_missing_pollutants_is_graceful():
    payload = {"reading": {"city": "Delhi", "date": "2025-04-23"}, "history": []}
    r = requests.post(f"{API}/predict", json=payload, timeout=10)
    if r.status_code == 503:
        pytest.skip("Model not loaded")
    assert r.status_code == 200


def test_predict_bad_date_format_is_rejected():
    bad = _sample_reading()
    bad["reading"]["date"] = None
    r = requests.post(f"{API}/predict", json=bad, timeout=5)
    assert r.status_code == 422


def test_ground_truth_unknown_id_is_404():
    r = requests.post(
        f"{API}/ground-truth",
        json={"prediction_id": str(uuid.uuid4()), "actual_aqi": 100.0},
        timeout=5,
    )
    assert r.status_code == 404


def test_ground_truth_round_trip():
    pred = requests.post(f"{API}/predict", json=_sample_reading(), timeout=10)
    if pred.status_code == 503:
        pytest.skip("Model not loaded")
    pid = pred.json()["prediction_id"]

    gt = requests.post(
        f"{API}/ground-truth",
        json={"prediction_id": pid, "actual_aqi": 238.0},
        timeout=5,
    )
    assert gt.status_code == 200
    assert gt.json()["status"] == "recorded"


def test_metrics_endpoint_is_scrapeable():
    r = requests.get(f"{API}/metrics", timeout=5)
    assert r.status_code == 200
    assert "predictions_total" in r.text


def test_stats_includes_trigger_config():
    r = requests.get(f"{API}/stats", timeout=5)
    assert r.status_code == 200
    body = r.json()
    for key in (
        "feedback_count_window",
        "feedback_count_threshold",
        "rmse_threshold",
        "window_hours",
        "count_gate_met",
        "rmse_gate_met",
    ):
        assert key in body, f"missing {key}"


def test_feedback_list_shape():
    r = requests.get(f"{API}/feedback?limit=10", timeout=5)
    assert r.status_code == 200
    body = r.json()
    assert "count" in body
    assert "rows" in body
    assert isinstance(body["rows"], list)


def test_feedback_csv_download():
    r = requests.get(f"{API}/feedback.csv?limit=10", timeout=5)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    first_line = r.text.split("\n", 1)[0]
    for col in ("prediction_id", "predicted_aqi", "actual_aqi", "abs_error"):
        assert col in first_line
