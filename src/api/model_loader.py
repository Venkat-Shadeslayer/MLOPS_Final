"""
Loads the Production-stage model from MLflow registry on API startup.

Design:
  - Singleton pattern: one ModelLoader instance per process
  - Fails fast on startup if no model exists — /ready will report not-ready
  - /ready endpoint lets Kubernetes/Compose know when it's safe to route traffic
  - Model version is captured at load time and embedded in every response so
    callers can correlate predictions with specific model versions
"""
from __future__ import annotations

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from src.utils.config import mlflow_config
from src.utils.logging import get_logger

log = get_logger(__name__)


class ModelLoader:
    """Lazy-loaded holder for the Production model."""

    def __init__(self) -> None:
        self._model = None
        self._version: str | None = None
        self._family: str | None = None
        self._feature_cols: list[str] | None = None
        self._load_error: str | None = None

    def load(self) -> None:
        """Fetch latest Production model. Idempotent."""
        if self._model is not None:
            return
        try:
            mlflow.set_tracking_uri(mlflow_config.tracking_uri)
            client = MlflowClient()
            # Get the current Production version
            versions = client.get_latest_versions(
                mlflow_config.model_name, stages=[mlflow_config.model_stage]
            )
            if not versions:
                raise RuntimeError(
                    f"No model registered under '{mlflow_config.model_name}' "
                    f"at stage '{mlflow_config.model_stage}'"
                )
            mv = versions[0]
            self._version = mv.version
            # Inspect the originating run for the model family tag
            run = client.get_run(mv.run_id)
            self._family = run.data.tags.get("model_family", "unknown")

            model_uri = f"models:/{mlflow_config.model_name}/{mlflow_config.model_stage}"
            log.info("Loading model from %s (family=%s, version=%s)",
                     model_uri, self._family, self._version)
            self._model = mlflow.pyfunc.load_model(model_uri)
            log.info("Model loaded. Signature=%s", self._model.metadata.signature)

            # Cache the feature columns from the model signature
            sig = self._model.metadata.signature
            if sig is not None and sig.inputs is not None:
                self._feature_cols = [inp.name for inp in sig.inputs.inputs]
                log.info("Feature columns (%d): first 5=%s",
                         len(self._feature_cols), self._feature_cols[:5])
        except Exception as e:
            self._load_error = str(e)
            log.error("Failed to load model: %s", e)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def version(self) -> str | None:
        return self._version

    @property
    def family(self) -> str | None:
        return self._family

    @property
    def feature_cols(self) -> list[str] | None:
        return self._feature_cols

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def predict(self, df: pd.DataFrame) -> float:
        """Run inference on a single-row DataFrame."""
        if self._model is None:
            raise RuntimeError("Model not loaded")
        preds = self._model.predict(df)
        # pyfunc.predict returns array-like; take the first element
        return float(preds[0] if hasattr(preds, "__iter__") else preds)


# Singleton instance imported by main.py
model_loader = ModelLoader()