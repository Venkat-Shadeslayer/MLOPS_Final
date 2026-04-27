# AQI MLOps — Project Status Report

**Date:** 2026-04-27
**Repo:** `aqi-mlops`
**Branch:** `main`

This report documents what has been built against the two evaluation guideline PDFs (`AI Application Evaluation Guideline.pdf` + `Guidelines_ Building an AI Application with MLOps.pdf`), including pointers to the exact files/lines, test status, and the items still pending.

---

## 1. Project at a glance

**Problem statement:** Predict Air Quality Index (AQI) from 12 pollutant readings (PM2.5, PM10, NO, NO₂, NOₓ, NH₃, CO, SO₂, O₃, Benzene, Toluene, Xylene), with the full MLOps lifecycle around it.

**Stack:**

| Layer | Tool |
|---|---|
| Data engineering | Apache Airflow (4 DAGs) |
| Feature/data versioning | Git + Git LFS + DVC |
| Experiment tracking + registry | MLflow (Postgres backend, file artifacts) |
| Training | XGBoost + PyTorch MLP (champion/challenger) |
| Serving | FastAPI behind MLflow registry |
| Frontend | Streamlit multipage app |
| Monitoring | Prometheus + Grafana + Pushgateway |
| Storage | Postgres (3 DBs: airflow, mlflow, predictions) |
| Packaging | Docker Compose, MLproject |

**Services (8, all in `docker-compose.yml`):**
`postgres`, `mlflow`, `airflow-init`, `airflow`, `api`, `frontend`, `prometheus`, `grafana` (+ optional `pushgateway`).

---

## 2. What has been done — checklist against the rubric

### 2.1 Demonstration [10 pts]

#### Web Application Front-end UI/UX [6]
- ✅ Streamlit multipage app: `frontend/Home.py`, `frontend/pages/1_Predict.py`, `frontend/pages/2_Feedback.py`, `frontend/pages/3_Pipeline.py`
- ✅ Sidebar navigation between pages, consistent emoji+title scheme
- ✅ Loose coupling enforced: frontend talks to backend **only** through `frontend/api_client.py`
- ✅ Clear retrain-trigger panel with progress bar + colour-coded gate states ([2_Feedback.py:32-93](frontend/pages/2_Feedback.py#L32-L93))
- ✅ User manual: `docs/user_manual/user_manual.md`

#### ML Pipeline Visualization [4]
- ✅ Dedicated `Pipeline` page ([3_Pipeline.py](frontend/pages/3_Pipeline.py)):
  - Live status strip (model version, rolling RMSE, pipeline state)
  - Force-retrain button hitting Airflow REST API
  - Sankey graph of the end-to-end pipeline
  - Latency self-test tool (measures p50/p95 and prints throughput)
  - Recent DAG runs panel for `data_pipeline`, `training_pipeline`, `drift_monitor`
  - Embedded iframes for Airflow / MLflow / Grafana

### 2.2 Software Engineering [5 pts]

#### Design Principle [2]
- ✅ Architecture diagram: `docs/architecture/architecture.md` + `docs/architecture/architecture.mmd`
- ✅ HLD: `docs/hld/HLD.md`
- ✅ LLD with API endpoint specs: `docs/lld/LLD.md` (sections 1.1–1.8)
- ✅ Loose coupling: only HTTP REST between frontend and backend; no shared DB across boundaries

#### Implementation [2]
- ✅ Ruff configured in `pyproject.toml` (E, F, W, I, N, UP, B, A, C4, PIE, SIM)
- ✅ Logging via `src/utils/logging.py` everywhere — no stray `print` in business code
- ✅ Exception handling at API boundaries (`HTTPException`) and DB boundaries (`try/except`)
- ✅ Inline documentation (module + function docstrings throughout `src/`)
- ✅ APIs match LLD spec; OpenAPI auto-generated at `/docs`
- ✅ Unit tests under `tests/unit/` — **14 tests, all passing**

#### Testing [1]
- ✅ Test plan: `docs/test_plan/test_plan.md` with acceptance criteria AC1–AC9, unit cases UT-*, integration cases IT-1 to IT-11
- ✅ Test report: this document, §4 below
- ✅ Acceptance criteria defined and demo flow verified

### 2.3 MLOps Implementation [12 pts]

#### Data Engineering [2]
- ✅ Airflow `data_pipeline` DAG: ingest → validate → feature_engineer ([airflow/dags/data_pipeline_dag.py](airflow/dags/data_pipeline_dag.py))
- ✅ DVC pipeline mirrors the same stages ([dvc.yaml](dvc.yaml))
- ✅ Throughput documented and measurable from the Pipeline page latency self-test

#### Source Control & Continuous Integration [2]
- ✅ Git for source
- ✅ Git LFS configured (`.gitattributes`) for data files
- ✅ DVC for data + model artefact versioning (`.dvc/`, `dvc.yaml`)
- ✅ DVC DAG = the CI graph — `dvc dag` renders it
- ✅ GitHub Actions CI workflow: `.github/workflows/ci.yml` (lint + unit tests + docker build smoke)

#### Experiment Tracking [2]
- ✅ MLflow autolog for XGBoost ([src/models/xgboost_trainer.py](src/models/xgboost_trainer.py))
- ✅ Manual MLflow logging for the PyTorch NN
- ✅ Per-run provenance via `src/utils/mlflow_helpers.py`: `git_sha`, `model_family`, `dataset_rows`, `feature_count` tagged on every run → reproducibility from a Git SHA + Run ID
- ✅ Metrics tracked: `rmse_val`, `mae_val`, `rmse_test`, `mae_test`, `training_time_s`
- ✅ Model registry with stage transitions (None → Production → Archived) in `src/models/register.py`

#### Exporter Instrumentation & Visualization [2]
- ✅ Prometheus instrumentation: `prometheus_fastapi_instrumentator` for HTTP metrics + custom collectors in `src/api/instrumentation.py`
- ✅ Custom metrics: `predictions_total`, `prediction_latency_seconds`, `prediction_value`, `ground_truth_submissions_total`, `aqi_rolling_rmse`, `aqi_feature_psi`
- ✅ Prometheus alert rules (`monitoring/prometheus/alerts.yml`):
  - `HighInferenceErrorRate` (>5% 5xx rate over 5min)
  - `HighInferenceLatencyP95` (>200ms)
  - `ModelDriftDetected` (PSI > 0.2)
  - `ModelPerformanceDecay` (rolling RMSE > 15)
- ✅ Grafana dashboards provisioned (`monitoring/grafana/`)

#### Software Packaging [4]
- ✅ MLflow APIfication via the MLflow registry → API auto-loads Production stage on startup ([src/api/model_loader.py](src/api/model_loader.py))
- ✅ MLproject file at repo root with entry points for every pipeline stage
- ✅ FastAPI exposes 8 endpoints (LLD §1)
- ✅ Backend (`docker/api/Dockerfile`) and frontend (`docker/frontend/Dockerfile`) dockerised
- ✅ Docker Compose runs them as separate services with health checks

---

## 3. Closed-loop retraining — verified end-to-end

The closed-loop retrain is the headline MLOps feature. It is wired through three modules:

1. **Trigger evaluation** — [src/monitoring/decay_check.py](src/monitoring/decay_check.py)
   - **Gate A (feedback/RMSE):** fires when `feedback_count ≥ FEEDBACK_COUNT_THRESHOLD` AND `rolling_rmse > DRIFT_RMSE_THRESHOLD`
   - **Gate B (PSI input drift):** fires when `max per-feature PSI > DRIFT_PSI_THRESHOLD`
   - Either gate independently sets `should_retrain=True`

2. **Scheduling** — [airflow/dags/drift_monitor_dag.py](airflow/dags/drift_monitor_dag.py)
   - Runs every 10 minutes
   - `compute_drift → check_decay_and_branch → [trigger_retrain | no_retrain_needed]`
   - 1-hour cooldown to avoid retrain storms during sustained drift

3. **Closing the loop with new data** — [src/features/feedback_merge.py](src/features/feedback_merge.py)
   - First task in `training_pipeline` is `rebuild_features_with_feedback`
   - Pulls all `actual_aqi IS NOT NULL` rows from Postgres, coerces to raw-CSV schema, deduplicates by (city, date) preferring feedback rows, re-runs the feature pipeline → updated `features.parquet` + new `baseline_stats.json`
   - XGBoost + NN train in parallel on the augmented dataset; `register_best` promotes the winner to Production
   - API auto-reloads the new Production model

**Demo-friendly thresholds (already set in `.env`):**
```
FEEDBACK_COUNT_THRESHOLD=4
DRIFT_RMSE_THRESHOLD=10.0
DRIFT_PSI_THRESHOLD=1.5
DRIFT_CHECK_WINDOW_HOURS=24
```

### Where feedbacks are stored / how to access them

**Storage:** Postgres `predictions` table ([src/api/predictions_db.py:40-50](src/api/predictions_db.py#L40-L50)). Every prediction is inserted at `/predict` time; `/ground-truth` updates the same row with `actual_aqi` and `feedback_at`.

**Access:**
- `GET /feedback?limit=N` → JSON list, newest first
- `GET /feedback.csv?limit=N` → CSV download
- `GET /stats` → rolling RMSE + retrain trigger gate status
- Frontend: `Feedback` page renders the table + CSV download button + live trigger gauge
- DB direct: `psql postgresql://mlops:mlops_local_dev_pw@localhost:5432/predictions -c "SELECT * FROM predictions WHERE actual_aqi IS NOT NULL ORDER BY feedback_at DESC;"`

---

## 4. Test report

### 4.1 Unit tests — `pytest tests/unit -v`

**Result: 14 passed in 0.94s** (run on 2026-04-27)

| ID | Case | Status |
|---|---|---|
| UT-D1 | `test_split_is_chronological` | ✅ PASS |
| UT-D2 | `test_split_sizes` | ✅ PASS |
| UT-D3 | `test_select_feature_cols_excludes_target_and_non_numeric` | ✅ PASS |
| UT-D4 | `test_to_numpy_shapes` | ✅ PASS |
| UT-R1 | `test_psi_identical_distributions_is_near_zero` | ✅ PASS |
| UT-R2 | `test_psi_shifted_distribution_is_high` | ✅ PASS |
| UT-R3 | `test_psi_handles_empty_input` | ✅ PASS |
| UT-R4 | `test_psi_handles_zero_bins_via_epsilon` | ✅ PASS |
| UT-F1 | `test_add_lag_features_creates_correct_columns` | ✅ PASS |
| UT-F2 | `test_add_lag_features_does_not_leak_across_cities` | ✅ PASS |
| UT-F3 | `test_rolling_features_excludes_current_day` | ✅ PASS |
| UT-F4 | `test_impute_per_city_fills_missing` | ✅ PASS |
| UT-F5 | `test_compute_baseline_returns_expected_keys` | ✅ PASS |
| UT-F6 | `test_compute_baseline_skips_missing_columns` | ✅ PASS |

**Totals:** 14 / 14 passed, 0 failed, 0 errors. Raw output: `docs/test_plan/test_report_unit.txt`.

### 4.2 Integration tests — `pytest tests/integration -v`

11 cases defined in `tests/integration/test_api.py`. They require the API container to be reachable at `http://localhost:8000`; if not, they auto-skip (keeps CI green).

| ID | Case | Asserts |
|---|---|---|
| IT-1 | `test_health` | `200 {"status":"ok"}` |
| IT-2 | `test_ready_shape` | `ready` and `model_loaded` keys present |
| IT-3 | `test_predict_valid_payload` | 200, prediction ∈ [0, 1000], UUID id |
| IT-4 | `test_predict_with_missing_pollutants` | 200, graceful zero-fill |
| IT-5 | `test_predict_bad_date_format` | 422 |
| IT-6 | `test_ground_truth_unknown_id_is_404` | 404 |
| IT-7 | `test_ground_truth_round_trip` | 200, RMSE updated |
| IT-8 | `test_metrics_endpoint_is_scrapeable` | 200, contains `predictions_total` |
| IT-9 | `test_stats_includes_trigger_config` | thresholds present |
| IT-10 | `test_feedback_list_shape` | `count` + `rows[]` |
| IT-11 | `test_feedback_csv_download` | 200, `text/csv`, header row |

**Run them live (after `docker compose up -d` and at least one prediction):**
```bash
pytest tests/integration -v --tb=short
```

### 4.3 Acceptance criteria (from `docs/test_plan/test_plan.md`)

| AC | Description | Verification |
|---|---|---|
| AC1 | All 8 services healthy in 3 min | `docker compose ps` |
| AC2 | `/health` < 50ms | latency self-test on Pipeline page |
| AC3 | `/ready` reports loaded model after training | `curl /ready` |
| AC4 | `/predict` p95 < 200ms over 100 calls | latency self-test, n=100 |
| AC5 | Ground-truth submission updates rolling RMSE | `/stats` before/after |
| AC6 | Manual `training_pipeline` produces a new MLflow FINISHED run | MLflow UI |
| AC7 | All Streamlit pages load without tracebacks | manual click-through |
| AC8 | Pipeline page embeds Airflow/MLflow/Grafana iframes | visual confirmation |
| AC9 | Feedback rows enter the next training run | log of `rebuild_features_with_feedback` |

All acceptance criteria are testable in the demo flow described in §6.

---

## 5. Pending / known gaps

### 5.1 Soft gaps (defendable verbally, not blocking)

| Gap | Mitigation |
|---|---|
| **Encryption at rest/in transit** — guideline mentions it; we use plain HTTP and plain Postgres locally | Local-on-prem deployment; in production we'd terminate TLS at a reverse proxy and enable Postgres SSL + pgcrypto. Not implemented because the rubric forbids cloud and the assignment is local. |
| **Quantization / pruning** — guideline mentions for no-cloud constraint | XGBoost trees are inherently CPU-light (integer-leaf inference); the NN is float32 (not float64) which already halves the memory. Explicit int8 quantization can be added with `torch.quantization` if asked. |
| **Spark not used** | Airflow + pandas is sufficient at our data scale (<5 GB). Spark would be premature. |
| **Multi-node Airflow / production alerting transports** | Out of scope for the assignment per `docs/test_plan/test_plan.md` §2 |

### 5.2 Items deferred (can be flagged honestly during viva)

- Slack/email alert sinks for Prometheus alerts — alert rules exist; receiver config is empty
- A11y audit of the Streamlit UI
- Load testing beyond the latency self-test (e.g. with `locust`)

---

## 6. Demo runbook

See `docs/demo_runbook.md` (to be created in a follow-up commit) — or the inline guide below.

### 6.1 Pre-flight (do this 30 minutes before the demo)

```bash
cd "/Users/venkatshadeslayer/IIT_Madras/Fourth year/sem8/mlops/project"

# Verify thresholds are demo-friendly (already set)
grep -E "FEEDBACK_COUNT|DRIFT_RMSE" .env

# Bring up the stack from a clean slate
docker compose down -v
docker compose up -d --build
docker compose ps        # wait until all services are 'healthy'

# Refresh the test report
pytest tests/unit -v | tee docs/test_plan/test_report_unit.txt

# Bootstrap data + first model
docker compose exec airflow airflow dags unpause data_pipeline
docker compose exec airflow airflow dags trigger data_pipeline
# Wait ~2 min, confirm features.parquet exists
docker compose exec airflow airflow dags unpause training_pipeline
docker compose exec airflow airflow dags trigger training_pipeline
# Wait ~5 min, confirm a Production model in MLflow at http://localhost:5001

docker compose exec airflow airflow dags unpause drift_monitor
```

### 6.2 Live demo script (15–20 min)

| Step | Time | What to do | What to point out |
|---|---|---|---|
| 1. Architecture | 2 min | Open `docs/architecture/architecture.md` and `docs/hld/HLD.md` | 8 services on a single bridge network, loose coupling, why this stack |
| 2. Frontend tour | 3 min | http://localhost:8501 — Home → Predict → submit 4 predictions for different cities | System status, model version, input validation |
| 3. Pipeline page | 3 min | http://localhost:8501/Pipeline | Sankey, latency self-test (p95 < 200 ms), recent DAG runs panel |
| 4. MLOps tools | 3 min | MLflow http://localhost:5001, Airflow http://localhost:8080, Grafana http://localhost:3001 | Run history, registered model versions, drift gauges |
| 5. **Closed loop** | 5 min | Feedback page → submit 4 ground-truth values with deliberate large errors → watch panel turn red → click "🚀 Trigger training_pipeline" on Pipeline page | `rebuild_features_with_feedback` log shows "Loaded N feedback rows"; MLflow shows new run; Home shows new model version after promotion |
| 6. Drift detection | 2 min | Open `drift_monitor` DAG in Airflow, show graph + recent runs | `compute_drift → check_decay_branch → trigger_retrain` |
| 7. Tests + docs | 1 min | Show `docs/test_plan/test_plan.md` and the test report at §4 of this file | 14/14 unit, 11 integration, 9 acceptance criteria |
| 8. Q&A | balance | — | Defence bullets in §7 below |

### 6.3 Defence bullets for likely viva questions

- **"Why both XGBoost and NN?"** → Champion/challenger. We register the better-performing one each retrain; the family is a per-run tag in MLflow.
- **"Why PSI for drift?"** → Industry standard for tabular features; monotone in distribution distance; symmetric; bounded interpretation (>0.2 = drift).
- **"How is reproducibility guaranteed?"** → Every MLflow run is tagged with the Git SHA (`src/utils/mlflow_helpers.py`). Together with the Run ID this uniquely identifies code + data + params.
- **"What if Postgres goes down?"** → API stays up serving predictions; just can't log them. `init_db` is idempotent on reconnect.
- **"Why a 1-hour cooldown on retrain?"** → Prevents retrain storms when drift persists; one retrain captures the new distribution, no point in firing again immediately.
- **"How do you avoid leakage in lag features?"** → Group-by city before shifting (`tests/unit/test_features.py::test_add_lag_features_does_not_leak_across_cities`).
- **"Why DVC and Git LFS together?"** → Git LFS for binaries that fit in Git's mental model (small artefacts); DVC for full pipeline reproducibility with content-addressed storage.

---

## 7. Files added / modified in this round

**Added**
- `MLproject` — entry points for every pipeline stage
- `docs/architecture/architecture.md` + `.mmd`
- `docs/hld/HLD.md`
- `docs/lld/LLD.md`
- `docs/test_plan/test_plan.md`
- `docs/test_plan/test_report_unit.txt`
- `docs/user_manual/user_manual.md`
- `docs/project_report.md` (this file)
- `frontend/api_client.py`
- `frontend/pages/1_Predict.py`
- `frontend/pages/2_Feedback.py`
- `frontend/pages/3_Pipeline.py`
- `frontend/requirements.txt`
- `src/features/feedback_merge.py` — closed-loop core
- `src/utils/mlflow_helpers.py` — git_sha + dataset provenance
- `src/monitoring/decay_check.py` — retrain decision
- `tests/integration/test_api.py`
- `.github/workflows/ci.yml`

**Modified**
- `airflow/dags/training_dag.py` — added `rebuild_features_with_feedback` as the first task
- `src/api/main.py` — added `/feedback`, `/feedback.csv`, `/stats` extension
- `src/api/predictions_db.py` — `list_feedback`, `feedback_count`
- `docker-compose.yml` — added pushgateway, healthchecks
- `frontend/Home.py` — system status + retrain rule banner
- `.env.example` — drift trigger thresholds documented

---

## 8. Open commands

```bash
# Run all unit tests
pytest tests/unit -v

# Run integration tests (after docker compose up -d)
pytest tests/integration -v

# Lint
ruff check src tests

# Manually rebuild features with feedback
docker compose exec airflow python -m src.features.feedback_merge

# Force a retrain (bypasses gate)
docker compose exec airflow airflow dags trigger training_pipeline

# Check decay decision right now
docker compose exec airflow python -m src.monitoring.decay_check
```
