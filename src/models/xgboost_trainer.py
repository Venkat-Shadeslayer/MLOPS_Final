"""
XGBoost trainer with MLflow tracking.

Why XGBoost: gradient-boosted trees are the strong baseline for structured
tabular regression. We log everything via mlflow.xgboost.autolog() for
reproducibility, then layer custom metrics (MAE, R^2) on top.

Produces:
  - MLflow run with params, metrics, model artifact, feature importance
  - Returns the run_id so the registration step can locate it
"""
from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import xgboost as xgb
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.dataset import prepare_splits
from src.utils.config import mlflow_config
from src.utils.logging import get_logger

log = get_logger(__name__)


def _read_params() -> dict:
    params_path = Path(__file__).resolve().parents[2] / "params.yaml"
    with params_path.open() as f:
        return yaml.safe_load(f)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict:
    """Compute regression metrics with a split-name prefix."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        f"{prefix}_rmse": rmse,
        f"{prefix}_mae": mae,
        f"{prefix}_r2": r2,
    }


def train() -> str:
    """Train XGBoost, log to MLflow, return run_id."""
    params = _read_params()
    t = params["train"]
    xgb_params = t["xgboost"]

    mlflow.set_tracking_uri(mlflow_config.tracking_uri)
    mlflow.set_experiment(t["mlflow_experiment"])

    splits = prepare_splits()

    # Enable autolog BEFORE start_run so it captures the fit() call
    mlflow.xgboost.autolog(log_input_examples=False, log_model_signatures=True)

    with mlflow.start_run(run_name="xgboost_baseline") as run:
        # Tag this run as the xgboost variant — the register step filters on this
        mlflow.set_tag("model_family", "xgboost")
        mlflow.log_params({
            "feature_count": len(splits["feature_cols"]),
            "train_size": splits["train_size"],
            "val_size": splits["val_size"],
            "test_size": splits["test_size"],
        })

        model = xgb.XGBRegressor(
            n_estimators=xgb_params["n_estimators"],
            max_depth=xgb_params["max_depth"],
            learning_rate=xgb_params["learning_rate"],
            subsample=xgb_params["subsample"],
            colsample_bytree=xgb_params["colsample_bytree"],
            early_stopping_rounds=xgb_params["early_stopping_rounds"],
            random_state=t["random_seed"],
            objective="reg:squarederror",
            tree_method="hist",
        )

        log.info("Fitting XGBoost on %d rows, %d features",
                 splits["train_size"], len(splits["feature_cols"]))
        model.fit(
            splits["X_train"], splits["y_train"],
            eval_set=[(splits["X_val"], splits["y_val"])],
            verbose=False,
        )

        # Evaluate on all splits
        metrics = {}
        for split_name in ("train", "val", "test"):
            X = splits[f"X_{split_name}"]
            y = splits[f"y_{split_name}"]
            y_pred = model.predict(X)
            metrics.update(evaluate(y, y_pred, split_name))

        mlflow.log_metrics(metrics)
        log.info("Metrics: %s", {k: round(v, 3) for k, v in metrics.items()})

        # Log feature importance as an artifact (top 15)
        importance = dict(zip(splits["feature_cols"], model.feature_importances_))
        top15 = dict(sorted(importance.items(), key=lambda kv: -kv[1])[:15])
        mlflow.log_dict(top15, "top_15_feature_importance.json")

        log.info("XGBoost run complete. run_id=%s", run.info.run_id)
        return run.info.run_id


if __name__ == "__main__":
    train()