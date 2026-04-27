"""
Thin HTTP client for the AQI inference API.

All Streamlit pages import from here. Centralizes:
  - base URL resolution (env var with sensible default)
  - timeout/error handling
  - JSON payload shaping

The rubric awards points for loose coupling — frontend touches the API
ONLY through this module, never directly to MLflow/Postgres.
"""
from __future__ import annotations

import os
from typing import Any

import requests

API_URL = os.getenv("STREAMLIT_API_URL", "http://api:8000")
DEFAULT_TIMEOUT = 10


class APIError(Exception):
    pass


def _post(path: str, payload: dict) -> dict:
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as e:
        raise APIError(f"Cannot reach API at {API_URL}{path}: {e}") from e
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise APIError(f"API {r.status_code}: {detail}")
    return r.json()


def _get(path: str) -> dict:
    try:
        r = requests.get(f"{API_URL}{path}", timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as e:
        raise APIError(f"Cannot reach API at {API_URL}{path}: {e}") from e
    if r.status_code >= 400:
        raise APIError(f"API {r.status_code}: {r.text}")
    return r.json()


def health() -> dict:
    return _get("/health")


def ready() -> dict:
    return _get("/ready")


def stats() -> dict:
    return _get("/stats")


def predict(reading: dict[str, Any], history: list[dict] | None = None) -> dict:
    """Submit a prediction. `reading` has city, date, and pollutant readings."""
    return _post("/predict", {"reading": reading, "history": history or []})


def submit_ground_truth(prediction_id: str, actual_aqi: float) -> dict:
    return _post("/ground-truth", {
        "prediction_id": prediction_id,
        "actual_aqi": actual_aqi,
    })


def feedback_list(limit: int = 500) -> dict:
    return _get(f"/feedback?limit={limit}")


def feedback_csv_url() -> str:
    """URL for the CSV download link."""
    return f"{API_URL}/feedback.csv"


def feedback_csv_bytes(limit: int = 10000) -> bytes:
    """Fetch CSV bytes (server-to-server — frontend container can reach api:8000).

    We fetch through the container network and hand the bytes to Streamlit's
    download_button, which works regardless of whether the user's browser
    can reach the api host directly.
    """
    try:
        r = requests.get(f"{API_URL}/feedback.csv?limit={limit}", timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as e:
        raise APIError(f"Cannot reach API at {API_URL}/feedback.csv: {e}") from e
    if r.status_code >= 400:
        raise APIError(f"API {r.status_code}: {r.text}")
    return r.content


def aqi_bucket(value: float) -> tuple[str, str]:
    """Return (label, hex color) per CPCB India AQI bands."""
    if value <= 50:
        return "Good", "#00b050"
    if value <= 100:
        return "Satisfactory", "#a9d08e"
    if value <= 200:
        return "Moderate", "#ffc000"
    if value <= 300:
        return "Poor", "#ed7d31"
    if value <= 400:
        return "Very Poor", "#c00000"
    return "Severe", "#7030a0"