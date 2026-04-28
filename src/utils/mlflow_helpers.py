"""
Shared MLflow logging utilities.

Keeps the "what do we ALWAYS log per run" policy in one place — trainers
call `log_run_provenance()` at the top of their `with mlflow.start_run()`
block so every run has consistent tags for reproducibility.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import mlflow


def _git_sha() -> str:
    """Return the current Git commit SHA, falling back to env var or 'unknown'."""
    env_sha = os.getenv("GIT_SHA")
    if env_sha:
        return env_sha
    try:
        repo = Path(__file__).resolve().parents[2]
        out = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, PermissionError, OSError):
        return "unknown"


def log_run_provenance(
    *,
    model_family: str,
    dataset_rows: int,
    feature_count: int,
    extra_tags: dict | None = None,
) -> None:
    """Tag the active MLflow run with reproducibility + dataset provenance.

    Must be called inside an active `mlflow.start_run()` block.
    """
    tags = {
        "model_family": model_family,
        "git_sha": _git_sha(),
        "dataset_rows": str(dataset_rows),
        "feature_count": str(feature_count),
    }
    if extra_tags:
        tags.update({k: str(v) for k, v in extra_tags.items()})
    mlflow.set_tags(tags)
