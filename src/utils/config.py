"""
Centralized configuration for the AQI MLOps project.

Every module imports settings from here rather than reading os.environ directly.
Gives us a single place to document, validate, and swap behavior between
local / docker / test environments.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_CSV_PATH = RAW_DATA_DIR / "city_day.csv"
PROCESSED_PARQUET_PATH = PROCESSED_DATA_DIR / "features.parquet"
BASELINE_STATS_PATH = PROCESSED_DATA_DIR / "baseline_stats.json"

# ----------------------------------------------------------------------------
# Dataset schema — canonical order for model inputs
# ----------------------------------------------------------------------------
POLLUTANT_FEATURES: list[str] = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene",
]
TARGET_COLUMN = "AQI"
DATE_COLUMN = "Date"
CITY_COLUMN = "City"


# ----------------------------------------------------------------------------
# MLflow
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class MLflowConfig:
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name: str = os.getenv("MLFLOW_EXPERIMENT", "aqi_regression")
    model_name: str = os.getenv("MODEL_NAME", "aqi_regressor")
    model_stage: str = os.getenv("MODEL_STAGE", "Production")


# ----------------------------------------------------------------------------
# Predictions database
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class DatabaseConfig:
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    user: str = os.getenv("POSTGRES_USER", "mlops")
    password: str = os.getenv("POSTGRES_PASSWORD", "mlops_local_dev_pw")
    database: str = os.getenv("PREDICTIONS_DB", "predictions")

    @property
    def url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


# ----------------------------------------------------------------------------
# Drift / retraining thresholds
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class DriftConfig:
    rmse_threshold: float = float(os.getenv("DRIFT_RMSE_THRESHOLD", "15.0"))
    error_rate_threshold: float = float(os.getenv("DRIFT_ERROR_RATE_THRESHOLD", "0.05"))
    check_window_hours: int = int(os.getenv("DRIFT_CHECK_WINDOW_HOURS", "24"))
    psi_threshold: float = float(os.getenv("DRIFT_PSI_THRESHOLD", "0.2"))


# ----------------------------------------------------------------------------
# Training defaults
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainingConfig:
    test_size: float = 0.15
    val_size: float = 0.15
    random_seed: int = 42
    xgboost_params: dict = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "reg:squarederror",
        "tree_method": "hist",
    })
    nn_params: dict = field(default_factory=lambda: {
        "hidden_dims": [64, 32],
        "dropout": 0.2,
        "lr": 1e-3,
        "batch_size": 256,
        "epochs": 50,
        "early_stopping_patience": 5,
    })


# Public singletons
mlflow_config = MLflowConfig()
db_config = DatabaseConfig()
drift_config = DriftConfig()
training_config = TrainingConfig()