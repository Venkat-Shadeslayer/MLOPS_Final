"""
Data ingestion stage.

Strategy:
  1. If raw CSV already exists at the expected path, skip download (idempotent).
  2. Else, try Kaggle CLI download.
  3. If Kaggle creds aren't available, raise a clear error pointing at the
     manual fallback path.

Run via:
    python -m src.data.ingest
or by DVC stage `dvc repro ingest`.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from src.utils.config import RAW_CSV_PATH, RAW_DATA_DIR
from src.utils.logging import get_logger

log = get_logger(__name__)


def _read_params() -> dict:
    """Load ingest params from params.yaml (project root)."""
    params_path = Path(__file__).resolve().parents[2] / "params.yaml"
    with params_path.open() as f:
        return yaml.safe_load(f)["ingest"]


def _kaggle_available() -> bool:
    """Check if Kaggle CLI is installed and credentials are present."""
    if shutil.which("kaggle") is None:
        return False
    # Kaggle reads ~/.kaggle/kaggle.json by default
    cred_path = Path.home() / ".kaggle" / "kaggle.json"
    return cred_path.exists()


def download_via_kaggle(dataset: str, dest: Path) -> None:
    """Use kaggle CLI to download and unzip the dataset."""
    log.info("Downloading dataset '%s' via Kaggle CLI to %s", dataset, dest)
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(dest), "--unzip"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        log.error("Kaggle CLI failed: %s", result.stderr)
        raise RuntimeError(f"Kaggle download failed: {result.stderr}")
    log.info("Kaggle download succeeded")


def ingest() -> Path:
    """Main ingestion entrypoint. Returns the path to the raw CSV."""
    params = _read_params()
    raw_path = RAW_CSV_PATH
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.exists() and raw_path.stat().st_size > 0:
        log.info("Raw CSV already present at %s (size=%d bytes), skipping download.",
                 raw_path, raw_path.stat().st_size)
        return raw_path

    if _kaggle_available():
        download_via_kaggle(params["kaggle_dataset"], RAW_DATA_DIR)
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Kaggle download completed but expected file {raw_path} not found. "
                f"Check the dataset slug and that the CSV inside is named '{raw_path.name}'."
            )
    else:
        msg = (
            f"\nNo raw CSV at {raw_path} and Kaggle CLI/credentials unavailable.\n"
            f"Manual fix:\n"
            f"  1. Visit https://www.kaggle.com/datasets/{params['kaggle_dataset']}\n"
            f"  2. Download the dataset and extract '{params['raw_csv_filename']}'\n"
            f"  3. Move it to: {raw_path}\n"
        )
        log.error(msg)
        raise FileNotFoundError(msg)

    log.info("Ingestion complete. Raw CSV at %s (size=%d bytes)",
             raw_path, raw_path.stat().st_size)
    return raw_path


if __name__ == "__main__":
    try:
        ingest()
    except (FileNotFoundError, RuntimeError) as e:
        log.error(str(e))
        sys.exit(1)