"""
Shared dataset logic for all model trainers.

Key design decision: chronological split (not random).
AQI time series has temporal structure — random split would leak future
information into training. Oldest 70% train, next 15% val, last 15% test.

Exposes:
  - load_features()       : reads features.parquet
  - train_val_test_split(): chronological split
  - select_feature_cols() : drops target + non-model columns
  - to_tensors()          : for PyTorch path
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.utils.config import PROCESSED_PARQUET_PATH
from src.utils.logging import get_logger

log = get_logger(__name__)


def _read_params() -> dict:
    params_path = Path(__file__).resolve().parents[2] / "params.yaml"
    with params_path.open() as f:
        return yaml.safe_load(f)


def load_features(path: Path = PROCESSED_PARQUET_PATH) -> pd.DataFrame:
    """Load the feature parquet written by the data pipeline."""
    if not path.exists():
        raise FileNotFoundError(
            f"Features not found at {path}. Run the data pipeline first "
            f"(docker exec mlops_airflow python -m src.features.transform)"
        )
    df = pd.read_parquet(path)
    log.info("Loaded features: %d rows, %d cols", len(df), len(df.columns))
    return df


def select_feature_cols(df: pd.DataFrame, target: str, exclude: list[str]) -> list[str]:
    """Everything except target, exclude list, and non-numeric columns."""
    excluded = set(exclude) | {target}
    feature_cols = [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]
    return feature_cols


def train_val_test_split(
    df: pd.DataFrame,
    date_col: str = "Date",
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split. NOT random — preserves temporal order.

    Returns (train_df, val_df, test_df), sorted by date.
    """
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    n = len(df_sorted)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    n_train = n - n_val - n_test

    train_df = df_sorted.iloc[:n_train].reset_index(drop=True)
    val_df = df_sorted.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = df_sorted.iloc[n_train + n_val:].reset_index(drop=True)

    log.info(
        "Split: train=%d (%s→%s), val=%d (%s→%s), test=%d (%s→%s)",
        len(train_df), train_df[date_col].min(), train_df[date_col].max(),
        len(val_df), val_df[date_col].min(), val_df[date_col].max(),
        len(test_df), test_df[date_col].min(), test_df[date_col].max(),
    )
    return train_df, val_df, test_df


def to_numpy(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract (X, y) as float32 numpy arrays for each split."""
    return (
        train_df[feature_cols].to_numpy(dtype=np.float32),
        train_df[target_col].to_numpy(dtype=np.float32),
        val_df[feature_cols].to_numpy(dtype=np.float32),
        val_df[target_col].to_numpy(dtype=np.float32),
        test_df[feature_cols].to_numpy(dtype=np.float32),
        test_df[target_col].to_numpy(dtype=np.float32),
    )


def prepare_splits() -> dict:
    """
    One-call data preparation used by both trainers.

    Returns a dict with: X_train, y_train, X_val, y_val, X_test, y_test,
    feature_cols, train_size, val_size, test_size.
    """
    params = _read_params()
    t = params["train"]

    df = load_features()
    feature_cols = select_feature_cols(
        df, target=t["target_column"], exclude=t["exclude_columns"]
    )
    log.info("Using %d feature columns", len(feature_cols))

    train_df, val_df, test_df = train_val_test_split(
        df, val_size=t["val_size"], test_size=t["test_size"]
    )
    X_train, y_train, X_val, y_val, X_test, y_test = to_numpy(
        train_df, val_df, test_df, feature_cols, t["target_column"]
    )

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "feature_cols": feature_cols,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
    }
