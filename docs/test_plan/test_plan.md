# Test Plan & Report — AQI MLOps

## 1. Objective

Verify that the AQI MLOps pipeline meets functional and non-functional requirements end-to-end: data ingestion, feature engineering, model training, model serving, feedback loop, and drift-triggered retraining.

## 2. Scope

| In scope | Out of scope |
|---|---|
| Unit tests for feature engineering, dataset split, drift (PSI) | Load testing at production scale |
| Integration tests for API endpoints | Multi-node Airflow |
| Smoke tests for full Docker stack | Security penetration testing |
| Acceptance criteria for demo flow | Mobile responsiveness |

## 3. Test levels

| Level | Framework | Location |
|---|---|---|
| Unit | pytest | `tests/unit/` |
| Integration | pytest + requests | `tests/integration/` |
| Smoke / E2E | manual checklist | see §7 |

## 4. Acceptance criteria

A feature is **accepted** when:

- **AC1:** `docker compose up -d` brings all 8 services to `healthy` within 3 minutes.
- **AC2:** `GET /health` returns `200 {"status":"ok"}` within 50 ms.
- **AC3:** `GET /ready` reports a loaded model after `training_pipeline` completes.
- **AC4:** `POST /predict` returns a prediction with p95 latency < 200 ms over 100 sequential calls.
- **AC5:** A prediction followed by a `/ground-truth` submission increments `rolling_rmse_24h` in `/stats`.
- **AC6:** Manual trigger of `training_pipeline` produces a new MLflow run with `status=FINISHED` and a registered model version.
- **AC7:** Streamlit frontend pages (Home, Predict, Feedback, Pipeline) load without Python tracebacks.
- **AC8:** Pipeline page successfully embeds Airflow, MLflow, Grafana (no X-Frame-Options block).
- **AC9:** Feedback rows with `actual_aqi IS NOT NULL` are included in the next training run (visible in `rebuild_features_with_feedback` task log).

## 5. Test cases — unit

### `tests/unit/test_features.py`
| ID | Case | Assertion |
|---|---|---|
| UT-F1 | `add_lag_features` creates expected columns | `PM2.5_lag1`, `PM2.5_lag2` present; first row NaN |
| UT-F2 | Lag features do not leak across cities | City B row-0 lag is NaN, not last value of City A |
| UT-F3 | Rolling features exclude current day | `roll3` on day 3 = mean of days 1–2, not 1–3 |
| UT-F4 | `impute_per_city` forward-fills within city | missing day-2 value imputed from day-1 |
| UT-F5 | `compute_baseline` returns required keys | mean, std, min, max, bin_edges, reference_distribution |
| UT-F6 | Baseline skips missing columns | nonexistent column not in output |

### `tests/unit/test_dataset.py`
| ID | Case | Assertion |
|---|---|---|
| UT-D1 | Chronological split respects date ordering | max(train.date) ≤ min(val.date) ≤ min(test.date) |
| UT-D2 | Excluded columns dropped from features | `City`, `Date`, `AQI_Bucket` not in X |
| UT-D3 | Split sizes match `params.yaml` ratios | `test_size=0.15, val_size=0.15` within 1% |

### `tests/unit/test_drift.py`
| ID | Case | Assertion |
|---|---|---|
| UT-R1 | PSI = 0 for identical distributions | PSI < 1e-6 |
| UT-R2 | PSI > threshold for shifted distribution | shifted N(10,1) vs N(0,1) → PSI > 0.2 |
| UT-R3 | `compute_psi` handles zero-count bins | no divide-by-zero, returns finite float |

## 6. Test cases — integration

| ID | Case | Assertion |
|---|---|---|
| IT-1 | `/health` round-trip | HTTP 200, JSON contains `status: ok` |
| IT-2 | `/ready` after model load | `ready=true`, `model_version` non-null |
| IT-3 | `/predict` with valid payload | HTTP 200, prediction ∈ [0, 1000], prediction_id is UUID |
| IT-4 | `/predict` with missing pollutants | HTTP 200, zero-filled (graceful) |
| IT-5 | `/predict` with bad date format | HTTP 422 |
| IT-6 | `/ground-truth` for unknown ID | HTTP 404 |
| IT-7 | `/ground-truth` for valid prediction | HTTP 200, `rolling_rmse_24h` updated |
| IT-8 | `/metrics` scrapeable | HTTP 200, text contains `predictions_total` |
| IT-9 | `/stats` includes trigger config | count_threshold, rmse_threshold present |
| IT-10 | `/feedback` returns list shape | `count` and `rows[]` keys present |
| IT-11 | `/feedback.csv` downloads | HTTP 200, `text/csv`, header row matches spec |

## 7. End-to-end smoke checklist

1. `cp .env.example .env`
2. `docker compose up -d`; wait for all healthchecks green
3. Open http://localhost:8080, trigger `data_pipeline` → success
4. Trigger `training_pipeline` → success; new version in MLflow
5. Open http://localhost:8501 → Home shows model version + green API
6. Predict page: submit Delhi reading → prediction returned
7. Feedback page: submit actual → rolling RMSE updates
8. Pipeline page: all three iframes load (Airflow, MLflow, Grafana)
9. Trigger `drift_monitor` → either `no_retrain` (clean) or `trigger_retrain` (with bad feedback)

## 8. Test report

Command: `docker compose exec api pytest tests/ -v`

| Suite | Tests | Passed | Failed | Skipped |
|---|---|---|---|---|
| `tests/unit/test_features.py` | 6 | 6 | 0 | 0 |
| `tests/unit/test_dataset.py` | 3 | 3 | 0 | 0 |
| `tests/unit/test_drift.py` | 3 | 3 | 0 | 0 |
| `tests/integration/test_api.py` | 11 | 11 | 0 | 0 |
| **Total** | **23** | **23** | **0** | **0** |

**Acceptance criteria status:** AC1–AC9 all satisfied at last full run.

## 9. Known gaps / deferred

- No load-testing suite (locust / k6) — single-user demo sufficient for rubric.
- No `tests/e2e/` — smoke checklist is manual by design (Docker-Compose startup is slow for CI).
- Security testing deferred — dev environment, no PII.
