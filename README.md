# AQI MLOps — Air Quality Index Prediction with Full MLOps Lifecycle

End-to-end MLOps application predicting **Air Quality Index (AQI)** from 12 pollutant
measurements, with **closed-loop retraining** driven by user feedback and data drift.

Built for IIT Madras, Sem 8 (B.Tech) — covers the full AI product lifecycle locally
under the "no cloud" constraint: ingestion → validation → feature engineering →
training → registry → serving → monitoring → drift detection → retraining.

---

## Quick start (3 commands)

```bash
git clone <this repo> && cd project
cp .env.example .env
docker compose up -d --build
```

Wait ~2 minutes for all 8 services to be healthy. Then bootstrap the data + first
model:

```bash
docker compose exec airflow airflow dags unpause data_pipeline
docker compose exec airflow airflow dags trigger data_pipeline
# wait ~2 min, then:
docker compose exec airflow airflow dags unpause training_pipeline
docker compose exec airflow airflow dags trigger training_pipeline
# wait ~5 min, then:
docker compose restart api  # API picks up the new Production model
curl -s http://localhost:8000/ready
```

Open the frontend → http://localhost:8501

For an end-user walkthrough of the web console (Predict / Feedback / Pipeline pages) see [docs/user_manual.md](docs/user_manual.md).

---

## Service URLs

| Service     | URL                         | Login         |
|-------------|-----------------------------|---------------|
| Frontend    | http://localhost:8501       | —             |
| API (docs)  | http://localhost:8000/docs  | —             |
| Airflow     | http://localhost:8080       | admin / admin |
| MLflow      | http://localhost:5001       | —             |
| Grafana     | http://localhost:3001       | admin / admin |
| Prometheus  | http://localhost:9090       | —             |
| Pushgateway | http://localhost:9091       | —             |
| Postgres    | localhost:5432              | mlops / mlops_local_dev_pw |

---

## Stack

| Layer | Tool | Why |
|---|---|---|
| Orchestration | Apache Airflow (4 DAGs) | Schedules data pipeline, training, drift monitor, simulator |
| Experiment tracking + registry | MLflow | Tracks runs, registers winning model to `Production` stage |
| Model training | XGBoost + PyTorch MLP | Champion/challenger; better val_rmse wins |
| Model serving | FastAPI | 8 endpoints; `/health`, `/ready` probes; Postgres-backed prediction log |
| Frontend | Streamlit (multipage) | Home / Predict / Feedback / Pipeline |
| Monitoring | Prometheus + Grafana + Pushgateway | HTTP + ML-specific metrics, 4 alert rules, dashboards |
| Alerting | Mailtrap (sandbox SMTP) | Captures Airflow `email_on_failure` notifications |
| Data + pipeline versioning | DVC + Git + Git LFS | `dvc.yaml` mirrors training pipeline; content-addressed caching |
| Storage | Postgres (3 logical DBs) | Airflow metadata + MLflow tracking + predictions log |
| Packaging | Docker Compose (8 services), MLproject | Environment parity; entry points for every pipeline stage |
| CI | GitHub Actions | Ruff lint + pytest unit + docker build smoke |

---

## Repository layout

```
.
├── airflow/dags/          # 5 DAGs: data_pipeline, training_pipeline, drift_monitor,
│                          #         simulator_normal/drift, _smtp_test
├── docker/                # Service Dockerfiles (api, frontend, airflow, mlflow)
├── docs/                  # Architecture, HLD, LLD, test plan, project report
├── frontend/              # Streamlit pages + thin REST client
├── monitoring/            # Prometheus rules + Grafana dashboards (provisioned)
├── notebooks/             # EDA scratch (gitignored data)
├── scripts/               # init_databases.sh — bootstraps Airflow/MLflow/predictions DBs
├── src/
│   ├── api/               # FastAPI app, Pydantic schemas, model loader, predictions DB
│   ├── data/              # ingest.py + validate.py
│   ├── features/          # transform.py (lag/rolling), baseline_stats.py, feedback_merge.py
│   ├── models/            # xgboost_trainer.py, nn_trainer.py, register.py, dataset.py
│   ├── monitoring/        # drift.py (PSI), decay_check.py (gates), simulator.py
│   └── utils/             # config.py, logging.py, mlflow_helpers.py (git_sha tagging)
├── tests/
│   ├── unit/              # 14 unit tests (features, dataset, drift)
│   └── integration/       # 11 integration tests (live API)
├── docker-compose.yml     # 8 services, single bridge network
├── dvc.yaml               # Pipeline DAG: ingest → validate → feature_engineer → train → register
├── MLproject              # Entry points for every pipeline stage
├── params.yaml            # Hyperparameters (XGBoost, NN), feature config
└── pyproject.toml         # Ruff + pytest config
```

---

## Closed-loop retraining (the headline feature)

1. User submits ground-truth via `POST /ground-truth` → row inserted into Postgres
   `predictions` table with `actual_aqi`
2. Airflow `drift_monitor` DAG runs every 10 minutes:
   - `compute_drift` → PSI per pollutant feature pushed to Prometheus
   - `check_decay_branch` → evaluates two independent gates:
     - **Gate A:** `feedback_count ≥ FEEDBACK_COUNT_THRESHOLD` AND `rolling_rmse > DRIFT_RMSE_THRESHOLD`
     - **Gate B:** `max_psi > DRIFT_PSI_THRESHOLD`
   - If either gate trips → `TriggerDagRunOperator` fires `training_pipeline`
3. `training_pipeline`:
   - `rebuild_features_with_feedback` — pulls all `actual_aqi IS NOT NULL` rows from
     Postgres, merges into raw CSV, re-runs feature engineering
   - `train_xgboost` and `train_pytorch_nn` run in parallel
   - `register_best_model` — picks the lower `val_rmse` winner, promotes to
     `Production`, archives previous version
4. Cooldown: 1 hour (prevents retrain storms during sustained drift)

Demo-friendly thresholds in `.env`:
- `FEEDBACK_COUNT_THRESHOLD=4`
- `DRIFT_RMSE_THRESHOLD=10.0`
- `DRIFT_PSI_THRESHOLD=1.5`

---

## Documentation

| Doc | Path |
|---|---|
| Architecture diagram | [docs/architecture/architecture.md](docs/architecture/architecture.md) |
| High-Level Design | [docs/hld/HLD.md](docs/hld/HLD.md) |
| Low-Level Design + API spec | [docs/lld/LLD.md](docs/lld/LLD.md) |
| Test plan + cases | [docs/test_plan/test_plan.md](docs/test_plan/test_plan.md) |
| Test report (unit) | [docs/test_plan/test_report_unit.txt](docs/test_plan/test_report_unit.txt) |
| Submission report | [docs/report_submission.md](docs/report_submission.md) |
| **User manual (non-technical)** | [docs/user_manual.md](docs/user_manual.md) |

---

## Tests

```bash
# Unit (no stack required)
pytest tests/unit -v          # 14 tests, ~1 second

# Integration (stack must be up at localhost:8000)
pytest tests/integration -v   # 11 tests; auto-skip if API down

# Lint
ruff check src tests
```

---

## License

Academic project — IIT Madras Sem 8 BTech MLOps coursework.
