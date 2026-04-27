# High-Level Design — AQI MLOps

## 1. Problem statement

Predict Air Quality Index (AQI) for Indian cities from 12 pollutant measurements
(PM2.5, PM10, NO, NO₂, NOₓ, NH₃, CO, SO₂, O₃, Benzene, Toluene, Xylene) with a
full MLOps lifecycle: automated ingestion, training, deployment, monitoring, and
retraining driven by real-world feedback.

## 2. Success metrics

| Category | Metric | Target |
|---|---|---|
| ML | RMSE on held-out test | < 40 AQI units |
| ML | MAE on held-out test | < 25 AQI units |
| Business | `/predict` p95 latency | < 200 ms |
| Business | Model rollout after retrain | < 5 min |
| Ops | Drift detection cadence | ≤ 10 min |
| Ops | Closed-loop retrain trigger | automatic, threshold-based |

## 3. Design choices and rationale

### 3.1 Two model families (XGBoost + PyTorch NN)
**Choice:** train both in parallel every run, register the one with lower validation RMSE.
**Rationale:** assignment-graded diversity; lets the registry logic (`register.py`) demonstrate model comparison rather than always-register. Keeps the NN as a live backup even when XGBoost wins.

### 3.2 Airflow over Spark/Ray
**Choice:** Airflow LocalExecutor.
**Rationale:** dataset is ~30k rows — Spark is overkill. Airflow gives us first-class DAG orchestration, a UI for free, cooldowns via `DagRun.find`, and the TriggerDagRunOperator for cross-DAG closed loops. Ray was rejected because we don't need distributed training.

### 3.3 MLflow (tracking + registry) over a custom store
**Choice:** MLflow with Postgres backend + named-volume artifact store.
**Rationale:** covers both rubric items ("experiment tracking" and "APIification") with one tool; the Production stage transition is the clean API for the serving layer.

### 3.4 Streamlit over React/Flask+JS
**Choice:** Streamlit multipage (`pages/` convention).
**Rationale:** team is two data-people, not JS engineers. Streamlit's native `components.iframe` lets us embed Airflow/MLflow/Grafana consoles with zero glue code — directly addresses the rubric's "orchestrating visualizations across multiple tools" criterion.

### 3.5 Feedback merge at the raw-CSV stage (not post-features)
**Choice:** `feedback_merge.py` reconstructs rows in the raw CSV schema and re-runs `transform.py`.
**Rationale:** if we merged post-features, retraining would diverge from initial training (different feature versions). Merging at the CSV layer means the *same* feature pipeline handles both, and drift baselines are recomputed consistently.

### 3.6 Closed-loop triggers: count-AND-RMSE gate, plus PSI
**Choice:** retrain when either (a) **feedback count ≥ N AND rolling RMSE > T** within the window, or (b) max feature PSI > P. All three (`FEEDBACK_COUNT_THRESHOLD=10`, `DRIFT_RMSE_THRESHOLD=100`, `DRIFT_PSI_THRESHOLD=1.5`) are configurable in `.env`.
**Rationale:** pure RMSE-only is too jumpy — a single badly-labeled feedback row would trigger an expensive retrain. Pure count-only is naive — a thousand perfect-prediction feedbacks shouldn't retrigger. Requiring **both** count and RMSE captures "enough real-world observations AND the model is demonstrably wrong." PSI remains as an independent input-drift signal that fires even when no feedback exists.

### 3.7 Docker Compose, not Kubernetes
**Choice:** 8 services in one `docker-compose.yml`.
**Rationale:** "No Cloud" rubric constraint; k8s adds operational burden without demo benefit. Swarm-mode compatibility is preserved (single-file topology).

## 4. Loose coupling enforcement

- Frontend never imports `mlflow`, `sqlalchemy`, or any ML library.
- `frontend/api_client.py` is the **only** coupling point — a thin HTTP wrapper around 6 REST endpoints.
- `STREAMLIT_API_URL` env var makes the API host swappable (Docker network name `api:8000` in compose, `localhost:8000` for local dev).
- No shared database connections across frontend/backend.

## 5. Non-functional requirements

| Requirement | How satisfied |
|---|---|
| Reproducibility | Every run tagged with Git SHA + MLflow run_id; DVC pipeline caches stage outputs |
| Versioning | Git (code), DVC (data), MLflow (models) |
| Observability | Prometheus scrape + pushgateway, Grafana dashboards, FastAPI access logs, Airflow task logs |
| Health checks | `/health` (liveness), `/ready` (model loaded), Docker `healthcheck` on 6 services |
| Security | `.env` never committed (`.gitignore`), DB credentials in env vars only, Grafana admin password envvar |
| Automation | Data ingestion (Airflow), training (Airflow + drift-triggered), deployment (registry stage = Production) |

## 5a. Observability coverage (per component)

Honest mapping of which components expose Prometheus metrics vs which are observable only through their own UIs — the rubric asks for end-to-end monitoring, and this is the delineation.

| Component | Prometheus scrape | Own UI / logs | Notes |
|---|---|---|---|
| FastAPI (inference) | ✅ `/metrics` via `prometheus_fastapi_instrumentator` | JSON access logs | Request count, latency histogram, custom model_version gauge |
| Drift monitor DAG | ✅ via Pushgateway | Airflow task logs | Pushes `rolling_rmse`, `psi`, `feedback_count`, `retrain_triggered` |
| Postgres | ❌ | container logs | Not a direct failure mode for this demo; healthcheck covers liveness |
| MLflow | ❌ (no exporter) | MLflow UI (:5001) | Observed via UI embed in Pipeline page |
| Airflow | ❌ (statsd-exporter not wired) | Airflow UI (:8080) + "Recent DAG runs" strip on Pipeline page (via REST) | Known gap vs a production setup — called out in test_plan future work |
| Grafana | ❌ | Grafana UI (:3001) | Self-observability |

The Pipeline console page consolidates all three UIs behind iframes plus a live status strip, so a single tab gives the operator full visibility despite the heterogeneous metric sources.

## 6. Out of scope (explicit non-goals)

- Multi-region deployment
- A/B testing / canary rollouts (only one Production version at a time)
- Streaming inference (batch sensor data)
- Model explainability (SHAP) — listed as future work
- Encryption at rest (listed as future work; dev environment)
