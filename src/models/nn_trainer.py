"""
PyTorch small-NN trainer with MLflow tracking.

Architecture: 2-layer MLP with dropout. Deliberately small — this is a
student project on a laptop, and the point is comparison vs XGBoost, not
SOTA. Manual MLflow logging (no autolog) so you can see exactly what's
being recorded.

Includes early stopping to avoid overfitting.
"""
from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.models.dataset import prepare_splits
from src.utils.config import mlflow_config
from src.utils.logging import get_logger

log = get_logger(__name__)


def _read_params() -> dict:
    params_path = Path(__file__).resolve().parents[2] / "params.yaml"
    with params_path.open() as f:
        return yaml.safe_load(f)


class MLPRegressor(nn.Module):
    """Simple feed-forward regressor."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    """Run the model on a loader, return (mean_loss, y_true, y_pred)."""
    model.eval()
    preds, truths = [], []
    loss_sum = 0.0
    n = 0
    loss_fn = nn.MSELoss(reduction="sum")
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss_sum += loss_fn(out, yb).item()
            n += yb.numel()
            preds.append(out.cpu().numpy())
            truths.append(yb.cpu().numpy())
    mean_loss = loss_sum / max(n, 1)
    return mean_loss, np.concatenate(truths), np.concatenate(preds)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {f"{prefix}_rmse": rmse, f"{prefix}_mae": mae, f"{prefix}_r2": r2}


def train() -> str:
    """Train MLP, log to MLflow, return run_id."""
    params = _read_params()
    t = params["train"]
    nn_params = t["nn"]

    torch.manual_seed(t["random_seed"])
    np.random.seed(t["random_seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    splits = prepare_splits()

    # Standardize features for NN (trees don't need this, NNs do)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(splits["X_train"])
    X_val = scaler.transform(splits["X_val"])
    X_test = scaler.transform(splits["X_test"])

    # Tensors + loaders
    def make_loader(X, y, shuffle):
        ds = TensorDataset(
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
        )
        return DataLoader(ds, batch_size=nn_params["batch_size"], shuffle=shuffle)

    train_loader = make_loader(X_train, splits["y_train"], shuffle=True)
    val_loader = make_loader(X_val, splits["y_val"], shuffle=False)
    test_loader = make_loader(X_test, splits["y_test"], shuffle=False)

    mlflow.set_tracking_uri(mlflow_config.tracking_uri)
    mlflow.set_experiment(t["mlflow_experiment"])

    with mlflow.start_run(run_name="nn_baseline") as run:
        mlflow.set_tag("model_family", "pytorch_nn")
        mlflow.log_params({
            "feature_count": len(splits["feature_cols"]),
            "train_size": splits["train_size"],
            "val_size": splits["val_size"],
            "test_size": splits["test_size"],
            **{f"nn_{k}": v for k, v in nn_params.items()},
        })

        model = MLPRegressor(
            input_dim=X_train.shape[1],
            hidden_dims=nn_params["hidden_dims"],
            dropout=nn_params["dropout"],
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=nn_params["lr"])
        loss_fn = nn.MSELoss()

        best_val_rmse = float("inf")
        best_epoch = -1
        patience = nn_params["early_stopping_patience"]
        bad_epochs = 0

        for epoch in range(1, nn_params["epochs"] + 1):
            model.train()
            train_loss_sum = 0.0
            n = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item() * yb.numel()
                n += yb.numel()
            train_mse = train_loss_sum / max(n, 1)

            val_mse, y_val_true, y_val_pred = evaluate(model, val_loader, device)
            val_rmse = float(np.sqrt(val_mse))

            mlflow.log_metric("train_mse", train_mse, step=epoch)
            mlflow.log_metric("val_rmse", val_rmse, step=epoch)

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_epoch = epoch
                bad_epochs = 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    log.info("Early stopping at epoch %d (best epoch %d, val_rmse %.4f)",
                             epoch, best_epoch, best_val_rmse)
                    break

        # Restore best weights
        model.load_state_dict(best_state)
        mlflow.log_metric("best_epoch", best_epoch)

        # Final metrics on all splits
        _, y_tr_true, y_tr_pred = evaluate(model, train_loader, device)
        _, y_va_true, y_va_pred = evaluate(model, val_loader, device)
        _, y_te_true, y_te_pred = evaluate(model, test_loader, device)

        final_metrics = {}
        final_metrics.update(regression_metrics(y_tr_true, y_tr_pred, "train"))
        final_metrics.update(regression_metrics(y_va_true, y_va_pred, "val"))
        final_metrics.update(regression_metrics(y_te_true, y_te_pred, "test"))
        mlflow.log_metrics(final_metrics)
        log.info("Metrics: %s", {k: round(v, 3) for k, v in final_metrics.items()})

        # Log model
        mlflow.pytorch.log_model(model, artifact_path="model")

        # Log scaler as a pickle artifact — inference needs it
        import pickle
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(scaler, f)
            mlflow.log_artifact(f.name, artifact_path="preprocessing")

        log.info("NN run complete. run_id=%s", run.info.run_id)
        return run.info.run_id


if __name__ == "__main__":
    train()