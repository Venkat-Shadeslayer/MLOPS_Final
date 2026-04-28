# AQI MLOps — Screen-Record Demo Runbook

**Target length:** 15–18 minutes
**Audience:** Evaluators (technical) — assume they know MLOps but not your project
**Style:** Show ‒ tell ‒ defend. Every claim should be backed by a thing on screen.

---

## 0. Pre-flight (do this 30+ minutes before recording — NOT on camera)

### 0.1 Sanity check the host

```bash
# 1. Docker Desktop must be running — whale icon in menu bar
docker info | head -5
# 2. Repo
cd "/Users/venkatshadeslayer/IIT_Madras/Fourth year/sem8/mlops/project"
# 3. Verify demo-friendly thresholds
grep -E "FEEDBACK_COUNT|DRIFT_RMSE|DRIFT_PSI" .env
# Expected:
#   FEEDBACK_COUNT_THRESHOLD=4
#   DRIFT_RMSE_THRESHOLD=10.0
#   DRIFT_PSI_THRESHOLD=1.5
```

### 0.2 Clean rebuild + bring stack up

```bash
docker compose down -v             # nuke previous volumes for a clean state
docker compose up -d --build       # build images + start 8 services
docker compose ps                  # wait until every service is "healthy" (~2 min)
```

✅ **Expected:** all of `postgres`, `mlflow`, `airflow`, `api`, `frontend`, `prometheus`, `grafana`, `pushgateway` show `healthy` or `running` (`airflow-init` exits 0 — that's correct).

### 0.3 Bootstrap data + first model (MUST be done before recording)

```bash
# Unpause + trigger data pipeline
docker compose exec airflow airflow dags unpause data_pipeline
docker compose exec airflow airflow dags trigger data_pipeline

# Wait ~2 minutes — verify it succeeded
docker compose exec airflow airflow dags list-runs -d data_pipeline | head -5

# Trigger first training run
docker compose exec airflow airflow dags unpause training_pipeline
docker compose exec airflow airflow dags trigger training_pipeline

# Wait ~5 minutes — confirm a Production model exists
curl -s http://localhost:8000/ready | jq
# Expected: {"ready": true, "model_loaded": true, "model_name": "aqi_regressor", "model_version": "1"}

# Unpause drift monitor (will run every 10 min during demo)
docker compose exec airflow airflow dags unpause drift_monitor
```

✅ **Expected before recording:** `/ready` returns `ready=true` with a model version. MLflow at http://localhost:5001 shows 2 finished runs (xgboost + nn). Registry has `aqi_regressor` with one Production version.

### 0.4 Sanity-check the front

Open http://localhost:8501 — all four pages (Home, Predict, Feedback, Pipeline) should render with no Python tracebacks. If any tab shows red, fix it before recording.

### 0.5 Pre-position browser tabs

Open these in order so you can ⌘+Tab between them:

| Tab | URL |
|---|---|
| 1 | http://localhost:8501 (Streamlit) |
| 2 | http://localhost:5001 (MLflow) |
| 3 | http://localhost:8080 (Airflow — admin/admin) |
| 4 | http://localhost:3001 (Grafana) |
| 5 | http://localhost:9090 (Prometheus) |
| 6 | http://localhost:8000/docs (FastAPI Swagger) |
| 7 | VS Code with the repo open at `docs/architecture/architecture.md` |
| 8 | A terminal at the repo root |

---

## 1. Demo script

> **🎙️ Lines you say are quoted in green-ish.** Bold = action you take.

### [0:00–1:30] Intro + architecture (1.5 min)

🎙️ *"This is the AQI MLOps project. The problem is air-quality-index regression from 12 pollutant readings, but the assignment is really about wrapping a model in the full MLOps lifecycle. Let me show you the architecture first."*

**Switch to VS Code, open `docs/architecture/architecture.md`.**

🎙️ *"Eight Docker services on a single bridge network. Postgres holds three logical DBs — one for Airflow metadata, one for MLflow tracking, one for prediction logs. MLflow does experiment tracking and the model registry. Airflow runs four DAGs. FastAPI exposes inference. Streamlit is the user-facing console. Prometheus + Grafana close the monitoring loop. The key design rule is loose coupling: the frontend only ever talks to the backend through REST — there's a thin client at `frontend/api_client.py` that's the single seam between them."*

**Open `docs/hld/HLD.md` briefly, scroll through it.**

🎙️ *"The HLD documents the design choices and rationale; LLD has the API contracts; test plan has acceptance criteria; user manual is for non-technical users. All in `docs/`."*

✅ **What viewer should see:** architecture markdown with the service list and the data-flow diagram.

---

### [1:30–3:30] Frontend — Home + Predict (2 min)

**Switch to tab 1 (Streamlit Home).**

🎙️ *"This is what a non-technical user sees. The system status panel pulls from `/health`, `/ready`, and `/stats` — three separate API calls — and tells me the API is up, the model is loaded at version one, and the rolling RMSE in the last 24 hours."*

**Point at the auto-retrain banner.**

🎙️ *"This banner is the closed-loop state in plain English. Right now it's showing zero feedbacks against a threshold of four. When that threshold is met, the next drift-monitor run triggers a retrain automatically."*

**Click into the Predict page.**

🎙️ *"This is where a user gets a prediction. I'll pick Delhi, fill in some pollutants — PM2.5 of 110, PM10 180, NO 15, NO2 40, and so on. Submit."*

**Submit. Note the prediction_id, predicted AQI, and latency.**

🎙️ *"That came back in about 12 milliseconds. Predicted AQI 245. The prediction_id is a UUID — every prediction is logged to Postgres with this ID, which is the hook for the feedback loop."*

**Repeat with 3 more cities — Mumbai, Bangalore, Chennai — different pollutant values.**

🎙️ *"I'll make three more predictions for different cities, because I'll need these later for the feedback step."*

✅ **What viewer should see:** four successful predictions, each returning a prediction_id and latency under 50 ms. The session keeps a list of recent predictions in the sidebar.

---

### [3:30–5:30] Pipeline page (2 min)

**Click into the Pipeline page.**

🎙️ *"This is the rubric's required pipeline-management console. Top strip shows live state from the API. Then a Sankey diagram of the end-to-end pipeline — Kaggle raw goes through the data pipeline DAG into the feature store, training reads from there, registry serves the API, predictions feed Postgres, drift monitor reads Postgres and conditionally triggers retraining."*

**Click "▶️ Run latency self-test" with n=20.**

🎙️ *"This fires twenty predictions and reports p50, p95, mean, and throughput. The HLD says our business metric target is p95 under 200 ms."*

**Wait for results.**

🎙️ *"P95 is [whatever] milliseconds — well under the 200 ms target."*

**Scroll down to "Recent DAG runs".**

🎙️ *"This pulls live from the Airflow REST API. Three columns — data pipeline, training pipeline, drift monitor — with the last five runs and their states. This is the rubric's required errors/failures/successes console."*

**Scroll to the embedded tools tabs.**

🎙️ *"And rather than reinventing visualization, we orchestrate the existing best-of-breed tools — Airflow, MLflow, Grafana — embedded as iframes. The rubric explicitly allows this."*

**Click each tab briefly to show they actually load (no X-Frame-Options block).**

✅ **What viewer should see:** Sankey renders, latency self-test reports p95 < 200ms, recent DAG runs show green checkmarks, all three iframe tabs load.

---

### [5:30–8:30] MLOps tools deep-dive (3 min)

🎙️ *"Let me show each MLOps tool standalone so you can see what's actually being tracked."*

**Switch to tab 3 (Airflow).** Login: `admin` / `admin`.

🎙️ *"Four DAGs. Data pipeline does ingest, validate, feature-engineer. Training pipeline rebuilds features with feedback first, then trains XGBoost and the PyTorch NN in parallel, then registers the winner. Drift monitor runs every ten minutes."*

**Click `training_pipeline` → Graph view.**

🎙️ *"Notice `rebuild_features_with_feedback` is the first task — that's what closes the loop. It pulls every prediction with ground truth from Postgres, merges them into the raw CSV, and re-runs feature engineering. So when a retrain fires, it's training on actual user-supplied ground truth, not just the original Kaggle data."*

**Switch to tab 2 (MLflow).**

🎙️ *"MLflow tracking server. Two runs from the bootstrap training: xgboost and pytorch_nn. Click the xgboost run — you can see the metrics: rmse_val, mae_val, training_time_s. Parameters are autologged. And critically, every run is tagged with `git_sha`, `dataset_rows`, and `feature_count` — so any run is reproducible from a Git SHA plus the run ID, which is what the guideline explicitly requires."*

**Click "Models" → `aqi_regressor`.**

🎙️ *"The registry. One Production version right now. When training_pipeline finishes a new run that beats the current best on rmse_val, `register.py` promotes the new version and archives the old one. The API picks up the new version on its next reload."*

**Switch to tab 4 (Grafana).**

🎙️ *"Grafana dashboards are provisioned at startup from `monitoring/grafana/`. You're seeing requests-per-second, p95 latency, rolling RMSE, prediction value distribution, and per-feature PSI for drift. This is the near-real-time view the rubric asks for."*

**Switch to tab 5 (Prometheus) → Status → Rules.**

🎙️ *"Four alert rules — HighInferenceErrorRate at >5% 5xx over 5 minutes, HighInferenceLatencyP95 over 200ms, ModelDriftDetected at PSI > 0.2, ModelPerformanceDecay at RMSE > 15. The 5% rule is what the guideline mentions explicitly."*

**Quick query to demonstrate metrics flow:** type `predictions_total` in the Prometheus query bar → Execute.

🎙️ *"And there are the predictions we made earlier, broken down by model version and family."*

✅ **What viewer should see:** Airflow DAGs list, training_pipeline graph showing `rebuild_features_with_feedback` first, MLflow runs page with metrics + git_sha tag visible, Grafana dashboard with live data, Prometheus alert rules, `predictions_total` query returning a non-zero number.

---

### [8:30–13:30] **THE CLOSED LOOP** (5 min — the money shot)

🎙️ *"This is the heart of the project — the closed-loop retrain. I'll submit ground-truth feedback for the four predictions I made earlier, and watch the system trigger a retrain by itself."*

**Switch to tab 1 (Streamlit) → Feedback page.**

🎙️ *"Top of the page: retrain trigger panel. Live count: 0 of 4 feedbacks. RMSE gate currently green because we have no data yet. Both gates need to be met for retrain to fire — that's the rule stated right here in plain English."*

**Quick-fill panel — submit ground truth for the 4 session predictions, but with deliberately large errors.**

🎙️ *"I'll submit ground truth that's deliberately way off the predictions — say predicted 245, actual 50. That'll spike the rolling RMSE."*

For each of the 4 predictions, set actual_aqi very different from predicted (e.g., predicted 245 → actual 50; predicted 180 → actual 400). Click Submit on each.

**Watch the panel update after each submission.**

🎙️ *"Watch the count: one of four, two of four... and there it goes — four of four, count gate met. Rolling RMSE is now 180-something, well above the threshold of 10. Both gates are red. The banner now says: both gates met, the next drift-monitor run will trigger retraining."*

**Show the feedback history table further down the page.**

🎙️ *"And the feedback history is right here — every submission persisted in Postgres, queryable via `/feedback` and downloadable as CSV via `/feedback.csv`. Both endpoints documented in the LLD."*

**Click Download CSV briefly to show it works.**

🎙️ *"For audit and external analysis."*

🎙️ *"Now I could wait ten minutes for `drift_monitor` to fire on its own, but for the demo I'll force-trigger it from the Pipeline page."*

**Pipeline page → click "🚀 Trigger training_pipeline now".**

🎙️ *"That hits Airflow's REST API and starts the DAG. Run ID returned. Let me jump into Airflow to watch it."*

**Switch to Airflow → training_pipeline → latest run → Graph view.**

🎙️ *"First task: `rebuild_features_with_feedback`. Click on it, view the logs..."*

**Open task logs.**

🎙️ *"There it is — 'Loaded 4 feedback rows from predictions DB'. Those four rows we just submitted are now part of the training set. Then xgboost and the NN train in parallel, register_best picks the winner."*

**Wait for the run to finish (~3-4 min). Use this time to talk about the trigger logic:**

🎙️ *"While that's running — let me explain the trigger precisely. There are two independent gates in `decay_check.py`. Gate A is feedback-and-RMSE: at least N feedbacks in the window AND rolling RMSE above threshold. Gate B is PSI input drift: max per-feature PSI above 0.2. Either gate independently triggers retrain. There's also a one-hour cooldown to prevent retrain storms — if we just retrained, the next drift-monitor pass skips even if gates are met."*

**Once training_pipeline finishes:** switch to MLflow.

🎙️ *"New run, status FINISHED. Same git_sha tag — reproducibility guaranteed. Click Models — `aqi_regressor` now has version 2 in Production, version 1 archived."*

**Switch back to Streamlit Home.**

🎙️ *"And the Home page now reports model version 2. The API auto-reloaded the new model. We just closed the loop end-to-end without a single manual code change."*

✅ **What viewer should see (in order):**
- Feedback panel updating: count goes 0→1→2→3→4, both gates red after the 4th submission
- "🚀 Trigger" button returns success with a dag_run_id
- Airflow graph view with `rebuild_features_with_feedback` log line `Loaded 4 feedback rows`
- training_pipeline goes green end-to-end
- MLflow shows a new FINISHED run with the git_sha tag
- Registry: `aqi_regressor` version 2 in Production, version 1 archived
- Streamlit Home now reports `v2`

---

### [13:30–15:00] Tests + docs + DVC (1.5 min)

**Switch to terminal.**

```bash
pytest tests/unit -v
```

🎙️ *"Fourteen unit tests covering feature engineering — lag features don't leak across cities, rolling features exclude the current day; dataset splitting is chronological; PSI handles edge cases. All passing in under a second."*

**Show the integration test count.**

```bash
pytest tests/integration --collect-only -q
```

🎙️ *"Eleven integration tests against the live API — they auto-skip if the API is down so CI stays green. Let me run them now while the stack is up."*

```bash
pytest tests/integration -v
```

🎙️ *"All eleven pass."*

**Show DVC DAG.**

```bash
dvc dag
```

🎙️ *"DVC DAG — same shape as the Airflow training pipeline but stage-by-stage, with content-addressed caching. This is our reproducibility-as-CI: `dvc repro` re-runs only the stages whose inputs changed."*

**Open `docs/test_plan/test_plan.md`.**

🎙️ *"Test plan with nine acceptance criteria, AC1 through AC9, all covered by what we just demoed. Test report with pass/fail counts is in `docs/project_report.md` section 4."*

✅ **What viewer should see:** `14 passed` from unit tests, `11 passed` from integration, DVC DAG diagram, test plan markdown.

---

### [15:00–16:30] Wrap + Q&A defence (1.5 min)

🎙️ *"Quick recap. Eight services on Docker Compose. Airflow orchestrates four DAGs. MLflow tracks experiments and serves the registry. FastAPI is the inference layer. Streamlit is the UI. Prometheus + Grafana are the monitoring layer. DVC versions the data pipeline. The closed loop is real — feedback persisted, gates evaluated every ten minutes, training automatically rebuilds features with feedback and promotes the winner."*

🎙️ *"Things I'd defend if pushed: we don't use Spark because the dataset fits in memory and Airflow with pandas is sufficient — Spark would be premature. We don't run on a cloud because the rubric forbids it. We don't have TLS or pgcrypto because this is an on-prem demo; in production we'd terminate TLS at a reverse proxy and enable Postgres SSL. Quantization isn't done explicitly because XGBoost trees are already CPU-light and the NN is float32, which already halves memory."*

🎙️ *"Repo: README at the root, docs in `docs/`, code in `src/`. Thanks."*

---

## 2. Recovery cheatsheet (if something breaks live)

| Symptom | Fix |
|---|---|
| API `/ready` returns `ready=false` | `docker compose restart api` (waits ~10s for model load) |
| Streamlit page shows traceback | `docker compose logs frontend --tail 30` — usually a stale module import; `docker compose restart frontend` |
| Airflow DAG stuck in `queued` | `docker compose exec airflow airflow scheduler &` — scheduler probably crashed |
| MLflow shows no runs | Wait — autolog flushes on `mlflow.end_run()`, sometimes 5–10s after training task completes |
| Iframe tab shows blank | Not an X-Frame-Options issue — most often Grafana cookie scope; refresh the parent page |
| Grafana dashboard "no data" | Open Prometheus, query `predictions_total` — if zero, you haven't made any predictions yet |
| Force a retrain | Pipeline page → "🚀 Trigger training_pipeline now"; or `docker compose exec airflow airflow dags trigger training_pipeline` |
| Reset everything | `docker compose down -v && docker compose up -d --build` (rebootstrap data + first model after) |

---

## 3. What success looks like (checklist for re-watching the recording)

After recording, scrub through and confirm every item below is visible at least once:

- [ ] Architecture diagram on screen
- [ ] HLD opened briefly
- [ ] LLD page or Swagger UI shown
- [ ] Streamlit Home with system-status panel
- [ ] At least 4 successful `/predict` calls with prediction_ids visible
- [ ] Pipeline page Sankey + latency self-test result + Recent DAG runs panel
- [ ] All 3 embedded iframes (Airflow / MLflow / Grafana) load inside Streamlit
- [ ] Airflow native UI: DAGs list + training_pipeline graph view
- [ ] MLflow native UI: at least 2 runs visible + git_sha tag visible on a run
- [ ] MLflow Registry: aqi_regressor with at least one Production version
- [ ] Grafana dashboard with live panels (not "no data")
- [ ] Prometheus rules page showing 4 alert rules
- [ ] Prometheus query for `predictions_total` returns non-zero
- [ ] Feedback page: count progresses 0 → 4
- [ ] Feedback page: both gates flip to met after 4 submissions
- [ ] Force-retrain button returns a dag_run_id
- [ ] Airflow log of `rebuild_features_with_feedback` shows `Loaded 4 feedback rows`
- [ ] Training_pipeline ends in `success` state
- [ ] MLflow registry shows version 2 in Production after retrain
- [ ] Streamlit Home reports v2 after retrain
- [ ] `pytest tests/unit -v` shows `14 passed`
- [ ] `pytest tests/integration -v` shows `11 passed`
- [ ] `dvc dag` rendered in terminal
- [ ] Verbal coverage: every tool the rubric asks about (DVC, Airflow, MLflow, Prometheus, Grafana, FastAPI, Docker Compose, MLproject) is named and shown

---

## 4. Time budget summary

| Section | Duration | Cumulative |
|---|---|---|
| Intro + architecture | 1.5 min | 1:30 |
| Frontend Home + Predict | 2 min | 3:30 |
| Pipeline page | 2 min | 5:30 |
| MLOps tools deep-dive | 3 min | 8:30 |
| **Closed loop** | 5 min | 13:30 |
| Tests + docs + DVC | 1.5 min | 15:00 |
| Wrap + defence | 1.5 min | 16:30 |
| **Total** | **~17 min** | |

If running long, drop tools deep-dive to 2 min and wrap to 1 min — the closed loop is the section evaluators care about most; do not shortchange it.
