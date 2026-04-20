"""Unit tests for drift (PSI) math."""
from __future__ import annotations

import numpy as np
import pytest

from src.monitoring.drift import compute_psi


def test_psi_identical_distributions_is_near_zero():
    data = np.random.RandomState(42).normal(50, 10, 1000)
    bin_edges = np.linspace(data.min(), data.max(), 11).tolist()
    counts, _ = np.histogram(data, bins=bin_edges)
    ref_dist = (counts / counts.sum()).tolist()
    # Same data → PSI should be ~0
    psi = compute_psi(data, bin_edges, ref_dist)
    assert psi < 0.05, f"Expected low PSI, got {psi}"


def test_psi_shifted_distribution_is_high():
    rng = np.random.RandomState(42)
    ref_data = rng.normal(50, 10, 1000)
    bin_edges = np.linspace(ref_data.min() - 5, ref_data.max() + 5, 11).tolist()
    counts, _ = np.histogram(ref_data, bins=bin_edges)
    ref_dist = (counts / counts.sum()).tolist()
    # Completely shifted data
    shifted = rng.normal(100, 10, 1000)
    psi = compute_psi(shifted, bin_edges, ref_dist)
    assert psi > 0.5, f"Expected high PSI for shifted data, got {psi}"


def test_psi_handles_empty_input():
    psi = compute_psi(
        np.array([]),
        bin_edges=[0, 1, 2, 3],
        reference_dist=[0.3, 0.4, 0.3],
    )
    assert psi == 0.0


def test_psi_handles_zero_bins_via_epsilon():
    ref_dist = [0.5, 0.5, 0.0]
    bin_edges = [0, 10, 20, 30]
    # Current data lands entirely in the 3rd bin
    current = np.array([25, 26, 27, 28, 29])
    psi = compute_psi(current, bin_edges, ref_dist)
    # Should not be NaN/inf
    assert np.isfinite(psi), f"PSI should be finite, got {psi}"
    assert psi > 0, f"PSI should be positive for different distribution, got {psi}"