"""
Live-traffic simulator for demo / automated testing.

Strategy:
  1. Hold out last 15% of rows from the feature dataset (by date).
  2. Each simulator run picks the next unseen "day" of readings.
  3. For each reading: POST to /predict → record prediction_id.
  4. After a lag (default 5 minutes), POST the actual AQI to /ground-truth.

State is persisted in a Postgres row so successive runs advance the cursor.

This is what makes the closed loop visible during the demo — you don't have
to manually click buttons for the RMSE graph to move.
"""
from __future__ import annotations

import json
import os
import random
from datetime import datetime

import pandas as pd
import requests
from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from src.utils.config import (
    CITY_COLUMN,
    DATE_COLUMN,
    POLLUTANT_FEATURES,
    PROCESSED_PARQUET_PATH,
    TARGET_COLUMN,
    db_config,
)
from src.utils.logging import get_logger

log = get_logger(__name__)

API_URL = os.getenv("SIMULATOR_API_URL", "http://api:8000")
HOLDOUT_FRAC = 0.15
MAX_READINGS_PER_RUN = 20


_Base = declarative_base()


class SimulatorState(_Base):
    __tablename__ = "simulator_state"
    id = Column(Integer, primary_key=True, default=1)
    cursor_date = Column(String, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow)


def _engine():
    return create_engine(db_config.url, pool_pre_ping=True)


def _init_state() -> None:
    eng = _engine()
    _Base.metadata.create_all(eng)
    Session = sessionmaker(bind=eng)
    with Session() as s:
        if s.query(SimulatorState).count() == 0:
            s.add(SimulatorState(id=1, cursor_date=None))
            s.commit()


def _get_cursor() -> str | None:
    eng = _engine()
    Session = sessionmaker(bind=eng)
    with Session() as s:
        row = s.get(SimulatorState, 1)
        return row.cursor_date if row else None


def _set_cursor(date_str: str) -> None:
    eng = _engine()
    Session = sessionmaker(bind=eng)
    with Session() as s:
        row = s.get(SimulatorState, 1)
        row.cursor_date = date_str
        row.updated_at = datetime.utcnow()
        s.commit()


def load_holdout() -> pd.DataFrame:
    if not PROCESSED_PARQUET_PATH.exists():
        raise FileNotFoundError(f"Features not found at {PROCESSED_PARQUET_PATH}")
    df = pd.read_parquet(PROCESSED_PARQUET_PATH)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    cutoff_idx = int(len(df) * (1 - HOLDOUT_FRAC))
    holdout = df.iloc[cutoff_idx:].reset_index(drop=True)
    log.info("Holdout: %d rows (%s → %s)",
             len(holdout),
             holdout[DATE_COLUMN].min(),
             holdout[DATE_COLUMN].max())
    return holdout


def submit_prediction(reading_row: pd.Series) -> tuple[str, float]:
    """POST /predict. Returns (prediction_id, predicted_aqi)."""
    payload = {
        "reading": {
            "city": str(reading_row[CITY_COLUMN]),
            "date": str(reading_row[DATE_COLUMN].date()),
        },
        "history": [],
    }
    for col in POLLUTANT_FEATURES:
        if col in reading_row and pd.notna(reading_row[col]):
            payload["reading"][col] = float(reading_row[col])
    r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    r.raise_for_status()
    body = r.json()
    return body["prediction_id"], float(body["predicted_aqi"])


def submit_ground_truth(prediction_id: str, actual_aqi: float) -> None:
    r = requests.post(
        f"{API_URL}/ground-truth",
        json={"prediction_id": prediction_id, "actual_aqi": actual_aqi},
        timeout=10,
    )
    r.raise_for_status()


def run_one_batch(inject_drift: bool = False, max_rows: int = MAX_READINGS_PER_RUN) -> dict:
    """
    Advance the cursor, submit up to `max_rows` predictions and their ground truths.

    If inject_drift=True, perturb inputs and ground truth to simulate model decay —
    useful for forcing the retraining DAG to fire during a demo.
    """
    _init_state()
    holdout = load_holdout()
    cursor = _get_cursor()

    if cursor is None:
        next_rows = holdout.head(max_rows)
    else:
        cursor_date = pd.to_datetime(cursor)
        remaining = holdout[holdout[DATE_COLUMN] > cursor_date]
        if remaining.empty:
            log.info("Simulator exhausted holdout — nothing to submit.")
            return {"submitted": 0, "exhausted": True}
        next_rows = remaining.head(max_rows)

    submitted = 0
    errors = 0
    for _, row in next_rows.iterrows():
        try:
            reading = row.copy()
            actual = float(reading[TARGET_COLUMN])
            if inject_drift:
                # Perturb inputs upward (simulate a bad pollution episode);
                # perturb actuals DOWN so model looks increasingly wrong.
                for col in POLLUTANT_FEATURES:
                    if col in reading and pd.notna(reading[col]):
                        reading[col] = float(reading[col]) * (1.5 + random.random())
                actual = actual +300  # intentional mismatch

            pred_id, pred_aqi = submit_prediction(reading)
            submit_ground_truth(pred_id, actual)
            submitted += 1
        except Exception as e:
            log.error("Failed to submit row: %s", e)
            errors += 1

    last_date = next_rows[DATE_COLUMN].max()
    _set_cursor(str(last_date.date()))
    log.info("Batch done: submitted=%d errors=%d cursor_advanced_to=%s",
             submitted, errors, last_date.date())
    return {
        "submitted": submitted,
        "errors": errors,
        "cursor": str(last_date.date()),
        "exhausted": False,
    }


def reset_cursor() -> None:
    """Reset for a fresh demo run."""
    _init_state()
    _set_cursor(None)
    log.info("Simulator cursor reset.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--drift", action="store_true", help="Inject artificial drift")
    p.add_argument("--reset", action="store_true", help="Reset cursor")
    p.add_argument("--rows", type=int, default=MAX_READINGS_PER_RUN)
    args = p.parse_args()
    if args.reset:
        reset_cursor()
    else:
        print(json.dumps(run_one_batch(inject_drift=args.drift, max_rows=args.rows), indent=2))
