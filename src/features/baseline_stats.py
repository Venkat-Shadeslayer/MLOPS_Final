"""
Baseline statistics for drift detection.

Per the second PDF: "students must calculate the statistical baseline
(mean, variance, distribution) of features to be used later for drift detection."

Saves a JSON file with per-feature mean/std/quantiles/histogram-bins.
Day 5's drift detector compares incoming inference data against this baseline
using PSI (Population Stability Index).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


def compute_baseline(
    df: pd.DataFrame, columns: list[str], n_bins: int = 10
) -> dict:
    """Compute distribution statistics for each column."""
    stats: dict = {}
    for col in columns:
        if col not in df.columns:
            log.warning("Column '%s' not in dataframe, skipping", col)
            continue
        series = df[col].dropna()
        if len(series) == 0:
            continue
        # Quantile-based bin edges (handles skew better than equal-width)
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.unique(np.quantile(series, quantiles))
        # Compute reference distribution (% of rows in each bin)
        counts, _ = np.histogram(series, bins=bin_edges)
        ref_dist = (counts / counts.sum()).tolist() if counts.sum() > 0 else []
        stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
            "q25": float(series.quantile(0.25)),
            "q75": float(series.quantile(0.75)),
            "n_observations": int(len(series)),
            "bin_edges": bin_edges.tolist(),
            "reference_distribution": ref_dist,
        }
    return stats


def compute_and_save_baseline(
    df: pd.DataFrame,
    pollutant_cols: list[str],
    target_col: str,
    n_bins: int,
    output_path: Path,
) -> dict:
    """Compute and persist baseline stats."""
    cols = list(pollutant_cols)
    if target_col in df.columns:
        cols.append(target_col)

    baseline = compute_baseline(df, cols, n_bins=n_bins)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump({"feature_stats": baseline, "n_bins": n_bins}, f, indent=2)
    log.info("Saved baseline stats for %d features to %s", len(baseline), output_path)
    return baseline
