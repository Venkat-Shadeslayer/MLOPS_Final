"""
Data validation stage (rubric: "automated checks for schema consistency
and missing values during ingestion").

Produces a JSON report at data/processed/validation_report.json.
Fails (exit 1) if hard constraints violated; warns otherwise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.utils.config import PROCESSED_DATA_DIR, RAW_CSV_PATH
from src.utils.logging import get_logger

log = get_logger(__name__)

VALIDATION_REPORT_PATH = PROCESSED_DATA_DIR / "validation_report.json"


def _read_params() -> dict:
    params_path = Path(__file__).resolve().parents[2] / "params.yaml"
    with params_path.open() as f:
        return yaml.safe_load(f)["validate"]


def validate(df: pd.DataFrame, params: dict) -> dict:
    """
    Run all validation checks and return a structured report.

    Hard failures (raise): missing required columns, row count below min.
    Soft warnings (logged + reported): high-missing columns, type oddities.
    """
    report: dict = {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "errors": [],
        "warnings": [],
        "missing_pct_per_column": {},
    }

    # Required columns
    required = set(params["required_columns"])
    present = set(df.columns)
    missing_cols = required - present
    if missing_cols:
        msg = f"Missing required columns: {sorted(missing_cols)}"
        report["errors"].append(msg)

    # Row count
    if len(df) < params["min_rows_required"]:
        report["errors"].append(
            f"Row count {len(df)} below minimum {params['min_rows_required']}"
        )

    # Missing-value percentages
    for col in df.columns:
        pct_missing = float(df[col].isna().mean())
        report["missing_pct_per_column"][col] = round(pct_missing, 4)
        if pct_missing > params["max_missing_pct_per_column"]:
            report["warnings"].append(
                f"Column '{col}' is {pct_missing:.1%} missing "
                f"(threshold {params['max_missing_pct_per_column']:.0%})"
            )

    # Target column sanity
    target = params["target_column"]
    if target in df.columns:
        target_na = float(df[target].isna().mean())
        report["target_missing_pct"] = round(target_na, 4)
        if target_na > 0.3:
            report["warnings"].append(
                f"Target '{target}' is {target_na:.1%} missing — many rows will drop"
            )

    # Date parseability (no exception raised, just inspected)
    date_col = params["date_column"]
    if date_col in df.columns:
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        unparseable = int(parsed.isna().sum())
        report["unparseable_dates"] = unparseable
        if unparseable > 0:
            report["warnings"].append(
                f"{unparseable} unparseable values in '{date_col}'"
            )

    return report


def main() -> None:
    params = _read_params()

    if not RAW_CSV_PATH.exists():
        log.error("Raw CSV not found at %s. Run ingest stage first.", RAW_CSV_PATH)
        sys.exit(1)

    log.info("Reading %s", RAW_CSV_PATH)
    df = pd.read_csv(RAW_CSV_PATH)
    log.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    report = validate(df, params)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with VALIDATION_REPORT_PATH.open("w") as f:
        json.dump(report, f, indent=2)
    log.info("Wrote validation report to %s", VALIDATION_REPORT_PATH)

    if report["warnings"]:
        for w in report["warnings"]:
            log.warning(w)

    if report["errors"]:
        for e in report["errors"]:
            log.error(e)
        log.error("Validation FAILED with %d error(s).", len(report["errors"]))
        sys.exit(1)

    log.info("Validation PASSED. %d warning(s).", len(report["warnings"]))


if __name__ == "__main__":
    main()