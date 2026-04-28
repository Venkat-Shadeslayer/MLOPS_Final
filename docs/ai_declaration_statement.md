# AI Declaration Statement

**Project:** AQI MLOps — Air Quality Index Prediction with Full MLOps Lifecycle
**Course:** IIT Madras, B.Tech Sem 8 — MLOps
**Date:** 2026-04-28

## 1. Tools Used

**Claude (Anthropic):** Used for code assistance, debugging, scaffolding,
pipeline architecture guidance, documentation drafting, and report
structuring.

## 2. Prompts Used

The following prompts were used during the course of this assignment:

- "what's the right way to break this AQI MLOps assignment into stages so i
  can plan the whole pipeline end to end?"
- "what's the kaggle CLI command to pull the city_day air quality dataset?"
- "for DVC, is a local remote enough for this assignment or should i set up
  a gdrive remote?"
- "how should ingest.py be structured — does it write straight to
  data/raw/city_day.csv or stage it somewhere first?"
- "what kind of schema and missingness checks make sense in validate.py and
  what should the validation_report.json look like?"
- "for transform.py, what's a sensible order — imputation first, then lag
  features and rolling means, then target encoding for city?"
- "do i need a baseline_stats step at all, or can i compute drift baselines
  on the fly later?"
- "in the xgboost trainer, which MLflow tags are actually useful at
  registry time — model_family, git_sha, anything else?"
- "is a shallow PyTorch MLP a reasonable second family to keep alongside
  xgboost, or am i over-engineering this?"
- "what's the cleanest rule for register.py — pick lower val_rmse across
  both families and archive the previous Production version?"
- "which endpoints does FastAPI actually need for this rubric — is
  /health, /ready, /predict, /ground-truth, /stats, /feedback,
  /feedback.csv, /metrics enough?"
- "what's the cleanest way to keep ModelLoader as a singleton in FastAPI
  without lifespan getting weird?"
- "for the predictions table, is JSONB for input_features a good idea or
  should i flatten the columns?"
- "is PSI per feature the standard drift metric here, or should i be
  looking at KS or wasserstein instead?"
- "for the retrain trigger, is count-AND-RMSE plus an independent PSI gate
  too aggressive or about right?"
- "is it correct to merge feedback at the raw-CSV layer rather than
  post-features, so the same transform pipeline handles both?"
- "what's the right DAG shape for data_pipeline — ingest → validate →
  feature_engineer, or do i split feature engineering further?"
- "for training_pipeline, can train_xgboost and train_pytorch_nn run as
  parallel tasks before register_best_model?"
- "how do i wire TriggerDagRunOperator from drift_monitor into
  training_pipeline cleanly, and what's a reasonable cooldown?"
- "for the streamlit side, is one page per concern (Home / Predict /
  Feedback / Pipeline) the right split, or am i overdoing it?"
- "what services do i actually need in docker-compose for this rubric —
  api, frontend, airflow, mlflow, postgres, prometheus, grafana,
  pushgateway?"
- "the API container can't reach mlflow — what hostname and port should it
  use inside the compose network?"
- "MLflow artifact_root looks broken inside the container — is the named
  volume mount wrong?"
- "_git_sha is throwing PermissionError inside the airflow container — why
  does that happen and how should it fall back?"
- "Airflow email_on_failure isn't firing — what's the right SMTP config
  for mailtrap?"
- "is there a clean way to verify mailtrap actually receives Airflow alert
  emails without waiting for a real failure?"
- "with prometheus_fastapi_instrumentator, what custom gauges make sense
  on top of the defaults — rolling_rmse_24h, predictions_total by
  model_version?"
- "what should the prometheus alert rules look like for
  HighInferenceLatencyP95 and ModelDriftDetected?"
- "what's a sensible Grafana dashboard layout for this project —
  predictions/sec, latency, rolling RMSE, predictions by model version?"
- "for the unit tests, which behaviours in transform.py, drift.py, and
  dataset.py are worth pinning down?"
- "for integration tests against the live /predict and /ground-truth
  endpoints, is it ok to skip them when the API isn't up?"
- "what should an architecture diagram description and an HLD actually
  cover for this rubric — success metrics, design choices, NFRs,
  out-of-scope?"
- "for the LLD, how detailed should the endpoint specs and the postgres
  schema be?"
- "what does a good test_plan.md look like — unit, integration, and the
  retrain-loop end-to-end test?"
- "given this set of screenshots, how should the project submission
  report be structured so each figure maps cleanly to a rubric item?"
- "what's the simplest local way to convert a markdown report with all
  these screenshots into a PDF?"
- "for HLD/LLD flowcharts, what's the cleanest way to render mermaid
  diagrams as static images and embed them in the report and design docs?"
