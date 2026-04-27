# Architecture — AQI MLOps

## Block diagram

```
                          ┌───────────────────────┐
                          │   Kaggle (city_day)   │
                          └──────────┬────────────┘
                                     │ DVC-tracked ingest
                                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          AIRFLOW (scheduler + webserver)             │
│                                                                      │
│   data_pipeline DAG  ──►  training_pipeline DAG  ──►  register       │
│        │                    ▲                                        │
│        │                    │ TriggerDagRun                          │
│        ▼                    │                                        │
│   drift_monitor DAG (every 10 min) ────────────────────────────┐     │
│        │  PSI + rolling RMSE                                   │     │
│        └───────────────────────────────────────────────────────┘     │
└──────────────┬───────────────────────────┬───────────────────────────┘
               │ metrics                    │ artifacts + params
               ▼                            ▼
        ┌────────────┐              ┌───────────────┐
        │ Prometheus │              │    MLflow     │
        │ (scrape)   │              │ tracking + reg│
        └─────┬──────┘              └──────┬────────┘
              │                             │ load Production
              ▼                             ▼
        ┌───────────┐     HTTP REST    ┌──────────────┐
        │  Grafana  │◄─────────────────│   FastAPI    │
        │ dashboards│                  │ /predict     │
        └───────────┘                  │ /ground-truth│
              ▲                        │ /health /ready/stats/metrics│
              │ iframe                 └─────┬────────┘
              │                              │ JSONB log
              │                              ▼
        ┌─────┴────────────┐          ┌───────────────┐
        │    Streamlit     │◄─────────┤   Postgres    │
        │ Home / Predict / │  via     │ predictions   │
        │ Feedback / Pipe  │ api_client│ airflow_db    │
        └──────────────────┘          │ mlflow_db     │
                                      └───────────────┘
```

## Blocks

| Block | Container | Responsibility |
|---|---|---|
| **Postgres** | `mlops_postgres` | Metadata store: Airflow scheduler, MLflow tracking, API predictions log. Separate logical databases, one instance. |
| **MLflow** | `mlops_mlflow` | Experiment tracking + model registry. Artifacts in named volume `mlflow_artifacts` (shared with airflow + api). Stages: None → Production → Archived. |
| **Airflow** | `mlops_airflow` | Orchestrates three DAGs: `data_pipeline` (ingest→validate→features), `training_pipeline` (feedback-merge → [xgb ∥ nn] → register), `drift_monitor` (PSI + RMSE → branch → trigger retrain). LocalExecutor. |
| **FastAPI** | `mlops_api` | Model serving. Loads `aqi_regressor/Production` at startup; hot-reloads on `/predict` if missing. Persists every prediction to Postgres for feedback loop. Exposes `/metrics` for Prometheus. |
| **Streamlit** | `mlops_frontend` | User-facing multipage app. Uses `api_client.py` as the **only** coupling to the backend (loose coupling requirement). Embeds Airflow/MLflow/Grafana in iframes on Pipeline page. |
| **Prometheus** | `mlops_prometheus` | Scrapes `/metrics` from API every 15s. Additional ML-specific gauges (`ROLLING_RMSE`, `PREDICTION_VALUE`, drift PSI) via pushgateway. |
| **Grafana** | `mlops_grafana` | Dashboards. Anonymous Viewer role + `GF_SECURITY_ALLOW_EMBEDDING=true` so Streamlit iframes work. |
| **Pushgateway** | `mlops_pushgateway` | Receives PSI metrics from Airflow drift DAG (short-lived jobs → pushgateway, not scrape). |

## Data flow — closed loop

1. User submits pollutant readings on Streamlit Predict page.
2. Streamlit → `api_client.predict()` → FastAPI `/predict`.
3. FastAPI calls MLflow-loaded model, returns prediction + UUID.
4. Prediction persisted to `predictions` table with raw `input_features` JSONB.
5. User later reports actual AQI via Feedback page → FastAPI `/ground-truth` updates row.
6. `drift_monitor` DAG (every 10 min) computes rolling RMSE + PSI. If threshold breached → triggers `training_pipeline`.
7. `training_pipeline` first task `rebuild_features_with_feedback` merges feedback rows (raw-CSV-shaped) into the training set, re-engineers features.
8. XGBoost + NN train in parallel; `register_best` promotes the winning run to `Production`.
9. FastAPI picks up the new version on next `/predict` (lazy reload).

## Design rationale

- **Named volume `mlflow_artifacts`** shared between mlflow, airflow, api: registered models are written by airflow-run training, served by api — bind mounts would require identical UIDs, named volumes don't.
- **Loose coupling:** frontend only knows `http://api:8000`, configurable via `STREAMLIT_API_URL`. No MLflow/Postgres creds in frontend image.
- **Separate drift DAG from training DAG:** drift is a fast observability loop (10 min); training is heavy (minutes). Coupling them with TriggerDagRunOperator gives independent schedules and cooldowns.
- **Feedback rows → raw-CSV schema:** letting feedback re-enter at the pre-feature-engineering stage means the same `transform.py` is used for historic and live data — no schema drift between training and retraining.
