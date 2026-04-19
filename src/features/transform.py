"""
Feature engineering stage.

Transformations applied (in order):
  1. Parse dates, sort by (City, Date)
  2. Per-city forward-fill then median-impute pollutants
  3. Lag features: PM2.5_lag1, PM2.5_lag2, ... for each pollutant + lag
  4. Rolling means: PM2.5_roll3, PM2.5_roll7, ... per (City, pollutant, window)
  5. City encoding (target encoding with global mean fallback)
  6. Drop rows where target (AQI) is missing
  7. Drop initial rows per city that have NaN lag features

Reads:  data/raw/city_day.csv
Writes: data/processed/features.parquet
        data/processed/baseline_stats.json   (via baseline_stats.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

from src.features.baseline_stats import compute_and_save_baseline
from src.utils.config import (
    BASELINE_STATS_PATH,
    PROCESSED_DATA_DIR,
    PROCESSED_PARQUET_PATH,
    RAW_CSV_PATH,
)
from src.utils.logging import get_logger

log = get_logger(__name__)


def _read_params() -> dict:
    params_path = Path(__file__).resolve().parents[2] / "params.yaml"
    with params_path.open() as f:
        return yaml.safe_load(f)


def add_lag_features(
    df: pd.DataFrame, pollutants: list[str], lags: list[int], group_col: str
) -> pd.DataFrame:
    """Add t-N lag features per group (city)."""
    out = df.copy()
    for pol in pollutants:
        if pol not in df.columns:
            continue
        grouped = df.groupby(group_col, sort=False)[pol]
        for lag in lags:
            out[f"{pol}_lag{lag}"] = grouped.shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame, pollutants: list[str], windows: list[int], group_col: str
) -> pd.DataFrame:
    """Add rolling-mean features per group, shifted by 1 to avoid target leakage."""
    out = df.copy()
    for pol in pollutants:
        if pol not in df.columns:
            continue
        grouped = df.groupby(group_col, sort=False)[pol]
        for win in windows:
            # shift(1) ensures the rolling window does NOT include the current day
            out[f"{pol}_roll{win}"] = (
                grouped.shift(1).rolling(window=win, min_periods=1).mean()
                .reset_index(level=0, drop=True)
            )
    return out


def encode_city(df: pd.DataFrame, target_col: str, city_col: str) -> pd.DataFrame:
    """Target-encode city using mean AQI per city. Persists encoding map for inference."""
    out = df.copy()
    global_mean = out[target_col].mean()
    city_means = out.groupby(city_col)[target_col].mean()
    out["city_target_enc"] = out[city_col].map(city_means).fillna(global_mean)
    # Save the encoding map so inference can use it
    encoding_map = {
        "global_mean": float(global_mean),
        "city_means": city_means.to_dict(),
    }
    enc_path = PROCESSED_DATA_DIR / "city_encoding.yaml"
    with enc_path.open("w") as f:
        yaml.safe_dump(encoding_map, f)
    log.info("Saved city encoding map to %s", enc_path)
    return out


def impute_per_city(df: pd.DataFrame, pollutants: list[str], city_col: str) -> pd.DataFrame:
    """Forward-fill within city, then median-impute remaining."""
    out = df.copy()
    for pol in pollutants:
        if pol not in out.columns:
            continue
        out[pol] = out.groupby(city_col)[pol].transform(lambda s: s.ffill())
        out[pol] = out[pol].fillna(out[pol].median())
    return out


def transform(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Apply the full feature engineering pipeline."""
    feat = params["features"]
    val = params["validate"]

    target_col = val["target_column"]
    date_col = val["date_column"]
    city_col = val["city_column"]
    pollutants = feat["pollutant_columns"]

    # Parse and sort
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values([city_col, date_col]).reset_index(drop=True)
    log.info("After date parse + sort: %d rows", len(df))

    # Impute
    df = impute_per_city(df, pollutants, city_col)
    log.info("After per-city imputation: %d rows, missing total=%d",
             len(df), int(df[pollutants].isna().sum().sum()))

    # Lag + rolling
    df = add_lag_features(df, pollutants, feat["lag_days"], city_col)
    df = add_rolling_features(df, pollutants, feat["rolling_windows"], city_col)
    log.info("After lag + rolling: %d rows, %d cols", len(df), len(df.columns))

    # City encoding (only if target is present — this is training data)
    if target_col in df.columns:
        df = encode_city(df, target_col, city_col)

    # Drop rows where target is missing
    if target_col in df.columns:
        before = len(df)
        df = df.dropna(subset=[target_col])
        log.info("Dropped %d rows with missing target", before - len(df))

    # Drop rows with NaN lag features (initial rows per city)
    lag_cols = [c for c in df.columns if "_lag" in c]
    if lag_cols:
        before = len(df)
        df = df.dropna(subset=lag_cols)
        log.info("Dropped %d rows with NaN lag features", before - len(df))

    return df


def main() -> None:
    params = _read_params()

    if not RAW_CSV_PATH.exists():
        log.error("Raw CSV not found at %s. Run ingest stage first.", RAW_CSV_PATH)
        sys.exit(1)

    log.info("Reading %s", RAW_CSV_PATH)
    df = pd.read_csv(RAW_CSV_PATH)
    log.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    features_df = transform(df, params)
    log.info("Final feature frame: %d rows, %d cols", len(features_df), len(features_df.columns))

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(PROCESSED_PARQUET_PATH, index=False)
    log.info("Wrote features to %s", PROCESSED_PARQUET_PATH)

    # Compute baseline stats for drift detection
    compute_and_save_baseline(
        features_df,
        pollutant_cols=params["features"]["pollutant_columns"],
        target_col=params["validate"]["target_column"],
        n_bins=params["baseline"]["num_psi_bins"],
        output_path=BASELINE_STATS_PATH,
    )


if __name__ == "__main__":
    main()