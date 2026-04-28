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

- "get the full picture of the AQI MLOps assignment and help me plan the
  whole pipeline end to end".
- "give me the kaggle command to download the city_day air quality dataset".
- "dvc remote — should i use a local remote for this assignment or push to
  gdrive?"
- "write me the scaffolding for ingest.py that pulls from kaggle and writes
  to data/raw/city_day.csv"
- "write validate.py that does schema and missingness checks and emits a
  validation_report.json"
- "write transform.py that does imputation, lag features, rolling means, and
  target encoding for city"
- "write baseline_stats.py that stores per-feature means/variances for drift
  comparison later"
- "write xgboost_trainer.py that logs to MLflow with model_family and
  git_sha tags"
- "write nn_trainer.py — a small PyTorch MLP, also logged to MLflow"
- "write register.py that picks the lower val_rmse run across both families
  and promotes to Production, archives the previous version"
- "write the FastAPI main.py with /health, /ready, /predict, /ground-truth,
  /stats, /feedback, /feedback.csv, /metrics"
- "ModelLoader should be a singleton — show me the cleanest way to do that
  in FastAPI"
- "write predictions_db.py with SQLAlchemy ORM for the predictions table"
- "write drift.py that computes PSI per feature against the baseline"
- "write decay_check.py with the count-AND-RMSE gate plus PSI gate"
- "write feedback_merge.py that pulls actual_aqi rows from postgres and
  merges them back into the raw CSV before retraining"
- "write the data_pipeline DAG: ingest → validate → feature_engineer"
- "write the training_pipeline DAG: rebuild_features_with_feedback →
  [train_xgboost ∥ train_pytorch_nn] → register_best_model"
- "write the drift_monitor DAG with TriggerDagRunOperator and a 1-hour
  cooldown"
- "write the streamlit pages — Home, Predict, Feedback, Pipeline — with a
  thin api_client.py wrapper"
- "write docker-compose.yml with api, frontend, airflow, mlflow, postgres,
  prometheus, grafana, pushgateway"
- "API container can't reach mlflow — what's the right service name and
  port to use?"
- "MLflow artifact_root is broken inside the container — how do i fix the
  named volume mount?"
- "_git_sha is throwing PermissionError inside the airflow container — how
  do i make it fall back gracefully when git isn't available?"
- "Airflow email_on_failure isn't sending — how do i wire mailtrap as the
  SMTP backend?"
- "write a tiny _smtp_test DAG that intentionally raises so i can verify
  mailtrap actually receives the alert"
- "write prometheus_fastapi_instrumentator setup with custom gauges for
  rolling_rmse_24h and predictions_total by model_version"
- "write the prometheus alert rules for HighInferenceLatencyP95 and
  ModelDriftDetected"
- "write a Grafana dashboard JSON that shows predictions/sec, latency,
  rolling RMSE, and predictions by model version"
- "write the unit tests for transform.py, drift.py, and dataset.py"
- "write the integration tests that hit the live /predict and /ground-truth
  endpoints"
- "write the architecture diagram description and the HLD with success
  metrics, design choices, NFRs, and out-of-scope items"
- "write the LLD with full endpoint specs, postgres schema, mlflow
  registry rules, and module layout"
- "write the test_plan.md document covering unit, integration, and the
  retrain-loop end-to-end test"
- "draft the project submission report from this set of screenshots and
  map each figure to the rubric"
- "convert report_submission.md to PDF including all the screenshots"
- "generate an HLD flowchart and an LLD flowchart for the system and embed
  them as images in the report and in the design docs"
- "write a short user manual for the streamlit console covering startup,
  the four pages, feedback submission, and basic troubleshooting"
- "based on the fair-use code-of-conduct guidelines, generate an AI
  declaration statement in the same style as my previous project"
