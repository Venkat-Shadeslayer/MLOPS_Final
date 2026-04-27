"""
Closed-loop: merge ground-truth feedback rows into the training dataset.

Predictions that received a ground-truth `actual_aqi` via /ground-truth are
valuable new training examples — they are real observations on live data
distributions the original training set may not cover. This module pulls
those rows from the predictions DB, coerces them to the raw-CSV schema
(City, Date, 12 pollutants, AQI), concatenates with the base raw CSV, and
re-runs feature engineering to produce an updated features.parquet.

Called as the first task of the training DAG so XGBoost + NN both train
on data that includes user feedback.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from src.features.baseline_stats import compute_and_save_baseline
from src.features.transform import _read_params, transform
from src.utils.config import (
    BASELINE_STATS_PATH,
    CITY_COLUMN,
    DATE_COLUMN,
    POLLUTANT_FEATURES,
    PROCESSED_DATA_DIR,
    PROCESSED_PARQUET_PATH,
    RAW_CSV_PATH,
    TARGET_COLUMN,
)
from src.utils.logging import get_logger

log = get_logger(__name__)

# Keys used by the API's PollutantReading schema for city/date (lowercase)
_READING_CITY_KEY = "city"
_READING_DATE_KEY = "date"


def load_feedback_as_raw_rows() -> pd.DataFrame:
    """Return a DataFrame of feedback rows shaped like the raw CSV schema.

    Empty DataFrame (with correct columns) if there is no feedback yet.
    """
    from src.api.predictions_db import get_session  # local import — airflow-side lazy

    cols = [CITY_COLUMN, DATE_COLUMN, *POLLUTANT_FEATURES, TARGET_COLUMN]
    sql = text(
        """
        SELECT input_features, actual_aqi
        FROM predictions
        WHERE actual_aqi IS NOT NULL
        """
    )
    try:
        with get_session() as s:
            rows = s.execute(sql).fetchall()
    except Exception as e:
        log.warning("Could not read feedback rows (DB unavailable?): %s", e)
        return pd.DataFrame(columns=cols)

    if not rows:
        log.info("No feedback rows with ground truth yet")
        return pd.DataFrame(columns=cols)

    records = []
    for feat, actual in rows:
        if not isinstance(feat, dict):
            continue
        rec = {
            CITY_COLUMN: feat.get(_READING_CITY_KEY),
            DATE_COLUMN: feat.get(_READING_DATE_KEY),
            TARGET_COLUMN: float(actual),
        }
        for pol in POLLUTANT_FEATURES:
            rec[pol] = feat.get(pol)
        records.append(rec)

    df = pd.DataFrame.from_records(records, columns=cols)
    df = df.dropna(subset=[CITY_COLUMN, DATE_COLUMN])
    log.info("Loaded %d feedback rows from predictions DB", len(df))
    return df


def rebuild_features_with_feedback() -> None:
    """Airflow task entrypoint: raw CSV + feedback rows -> features.parquet.

    Feedback rows that collide with existing (City, Date) pairs replace the
    CSV row — the ground-truth observation is more authoritative than the
    original (which may have been missing AQI).
    """
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(f"Raw CSV missing at {RAW_CSV_PATH}. Run ingest first.")

    base = pd.read_csv(RAW_CSV_PATH)
    log.info("Base raw CSV: %d rows", len(base))

    feedback = load_feedback_as_raw_rows()

    if len(feedback):
        base[DATE_COLUMN] = pd.to_datetime(base[DATE_COLUMN], errors="coerce")
        feedback[DATE_COLUMN] = pd.to_datetime(feedback[DATE_COLUMN], errors="coerce")
        combined = pd.concat([base, feedback], ignore_index=True, sort=False)
        # keep='last' lets feedback override CSV rows with the same (city, date)
        before = len(combined)
        combined = combined.drop_duplicates(subset=[CITY_COLUMN, DATE_COLUMN], keep="last")
        log.info(
            "Merged %d feedback rows; %d duplicates collapsed; %d total rows",
            len(feedback), before - len(combined), len(combined),
        )
    else:
        combined = base

    params = _read_params()
    features_df = transform(combined, params)
    log.info("Feature frame: %d rows, %d cols", len(features_df), len(features_df.columns))

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(PROCESSED_PARQUET_PATH, index=False)
    log.info("Wrote %s", PROCESSED_PARQUET_PATH)

    compute_and_save_baseline(
        features_df,
        pollutant_cols=params["features"]["pollutant_columns"],
        target_col=params["validate"]["target_column"],
        n_bins=params["baseline"]["num_psi_bins"],
        output_path=BASELINE_STATS_PATH,
    )


if __name__ == "__main__":
    rebuild_features_with_feedback()
