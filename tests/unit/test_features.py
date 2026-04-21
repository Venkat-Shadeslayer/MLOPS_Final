"""Unit tests for feature engineering. Fast, no external deps."""
from __future__ import annotations

import pandas as pd
import pytest

from src.features.baseline_stats import compute_baseline
from src.features.transform import (
    add_lag_features,
    add_rolling_features,
    impute_per_city,
)


@pytest.fixture
def toy_df() -> pd.DataFrame:
    """Two cities, 5 days each, one pollutant."""
    return pd.DataFrame({
        "City": ["A"] * 5 + ["B"] * 5,
        "Date": pd.date_range("2024-01-01", periods=5).tolist() * 2,
        "PM2.5": [10.0, 20.0, 30.0, 40.0, 50.0,
                  100.0, 110.0, 120.0, 130.0, 140.0],
        "AQI":   [50, 60, 70, 80, 90, 200, 210, 220, 230, 240],
    })


def test_add_lag_features_creates_correct_columns(toy_df):
    out = add_lag_features(toy_df, ["PM2.5"], [1, 2], "City")
    assert "PM2.5_lag1" in out.columns
    assert "PM2.5_lag2" in out.columns
    # First row of city A: lag1 should be NaN
    a = out[out["City"] == "A"].reset_index(drop=True)
    assert pd.isna(a.loc[0, "PM2.5_lag1"])
    # Second row: lag1 should be 10.0
    assert a.loc[1, "PM2.5_lag1"] == 10.0


def test_add_lag_features_does_not_leak_across_cities(toy_df):
    out = add_lag_features(toy_df, ["PM2.5"], [1], "City")
    b = out[out["City"] == "B"].reset_index(drop=True)
    # First row of city B should NOT have city A's last value as a lag
    assert pd.isna(b.loc[0, "PM2.5_lag1"])


def test_rolling_features_excludes_current_day(toy_df):
    out = add_rolling_features(toy_df, ["PM2.5"], [3], "City")
    a = out[out["City"] == "A"].reset_index(drop=True)
    # Day 3 (index 2): rolling-3 of preceding values [10, 20] = 15.0
    # NOT 20.0 (which would be mean of [10,20,30] including current day)
    assert a.loc[2, "PM2.5_roll3"] == pytest.approx(15.0)


def test_impute_per_city_fills_missing(toy_df):
    df = toy_df.copy()
    df.loc[1, "PM2.5"] = None
    out = impute_per_city(df, ["PM2.5"], "City")
    # ffill should propagate prior value (10.0)
    assert out.loc[1, "PM2.5"] == 10.0


def test_compute_baseline_returns_expected_keys(toy_df):
    stats = compute_baseline(toy_df, ["PM2.5", "AQI"], n_bins=4)
    assert "PM2.5" in stats
    assert "AQI" in stats
    for col in ["mean", "std", "min", "max", "median",
                "q25", "q75", "bin_edges", "reference_distribution"]:
        assert col in stats["PM2.5"], f"Missing key {col}"


def test_compute_baseline_skips_missing_columns(toy_df):
    stats = compute_baseline(toy_df, ["PM2.5", "NONEXISTENT"], n_bins=4)
    assert "PM2.5" in stats
    assert "NONEXISTENT" not in stats
