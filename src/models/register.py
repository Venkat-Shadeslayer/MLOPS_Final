"""
Model registration.

Compares the most recent XGBoost run and the most recent NN run in the
configured experiment, picks the winner by validation RMSE, and
registers it under the configured registry name with stage=Production.

Viva-defensible: manual UI promotion breaks reproducibility guarantees.
By registering via code, every training run ends with a deterministic
model version bump tied to a specific Git commit + MLflow run.
"""
from __future__ import annotations

from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient

from src.utils.config import mlflow_config
from src.utils.logging import get_logger

log = get_logger(__name__)


def _read_params() -> dict:
    params_path = Path(__file__).resolve().parents[2] / "params.yaml"
    with params_path.open() as f:
        return yaml.safe_load(f)


def latest_run_for_family(client: MlflowClient, experiment_id: str, family: str):
    """Return the most recent run tagged model_family=<family>."""
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.model_family = '{family}' and attributes.status = 'FINISHED'",
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )
    return runs[0] if runs else None


def register_best() -> dict:
    """Main entrypoint. Returns summary dict."""
    params = _read_params()
    t = params["train"]
    registry_name = t["model_registry_name"]
    experiment_name = t["mlflow_experiment"]

    mlflow.set_tracking_uri(mlflow_config.tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment '{experiment_name}' not found.")

    xgb_run = latest_run_for_family(client, experiment.experiment_id, "xgboost")
    nn_run = latest_run_for_family(client, experiment.experiment_id, "pytorch_nn")

    candidates = []
    if xgb_run:
        candidates.append(("xgboost", xgb_run))
    if nn_run:
        candidates.append(("pytorch_nn", nn_run))

    if not candidates:
        raise RuntimeError("No candidate runs found — train something first.")

    # Pick by lowest val_rmse
    def val_rmse(run) -> float:
        return run.data.metrics.get("val_rmse", float("inf"))

    family, best = min(candidates, key=lambda fam_run: val_rmse(fam_run[1]))
    log.info("Winner: %s (run_id=%s, val_rmse=%.4f)",
             family, best.info.run_id, val_rmse(best))

    # Register. Path to the logged model artifact is "model" (both trainers use that)
    model_uri = f"runs:/{best.info.run_id}/model"
    registered = mlflow.register_model(model_uri=model_uri, name=registry_name)
    log.info("Registered %s version %s", registry_name, registered.version)

    # Promote to Production, archive others
    client.transition_model_version_stage(
        name=registry_name,
        version=registered.version,
        stage="Production",
        archive_existing_versions=True,
    )
    log.info("Promoted %s v%s to Production", registry_name, registered.version)

    return {
        "winning_family": family,
        "run_id": best.info.run_id,
        "val_rmse": val_rmse(best),
        "registry_name": registry_name,
        "version": registered.version,
    }


if __name__ == "__main__":
    summary = register_best()
    print(summary)