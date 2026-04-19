"""Unit tests for dataset split logic. No MLflow, no network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.dataset import (
    select_feature_cols,
    to_numpy,
    train_val_test_split,
)


@pytest.fixture
def toy_df() -> pd.DataFrame:
    """100 rows with a Date column, target, and some features."""
    return pd.DataFrame({
        "City": ["A"] * 100,
        "Date": pd.date_range("2023-01-01", periods=100, freq="D"),
        "AQI": np.arange(100, dtype=float),
        "PM2.5": np.random.rand(100),
        "PM10": np.random.rand(100),
        "AQI_Bucket": ["Moderate"] * 100,  # non-numeric, must be excluded
    })


def test_split_is_chronological(toy_df):
    train, val, test = train_val_test_split(toy_df, val_size=0.2, test_size=0.2)
    assert train["Date"].max() < val["Date"].min()
    assert val["Date"].max() < test["Date"].min()


def test_split_sizes(toy_df):
    train, val, test = train_val_test_split(toy_df, val_size=0.2, test_size=0.2)
    assert len(val) == 20
    assert len(test) == 20
    assert len(train) == 60


def test_select_feature_cols_excludes_target_and_non_numeric(toy_df):
    cols = select_feature_cols(toy_df, target="AQI",
                                exclude=["City", "Date", "AQI_Bucket"])
    assert "AQI" not in cols
    assert "City" not in cols
    assert "Date" not in cols
    assert "AQI_Bucket" not in cols
    assert "PM2.5" in cols
    assert "PM10" in cols


def test_to_numpy_shapes(toy_df):
    train, val, test = train_val_test_split(toy_df, val_size=0.2, test_size=0.2)
    cols = ["PM2.5", "PM10"]
    X_tr, y_tr, X_v, y_v, X_te, y_te = to_numpy(train, val, test, cols, "AQI")
    assert X_tr.shape == (60, 2)
    assert X_v.shape == (20, 2)
    assert X_te.shape == (20, 2)
    assert y_tr.shape == (60,)
    assert X_tr.dtype == np.float32