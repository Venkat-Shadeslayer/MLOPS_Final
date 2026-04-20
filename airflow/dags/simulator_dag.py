"""
Live-traffic simulator DAG.

Every 5 minutes: advance the simulator cursor by ~20 readings, submit them
to /predict and /ground-truth. This produces the continuous traffic that
drives the demo.

Two variants:
  - simulator_normal: clean data, RMSE stays low
  - simulator_drift:  perturbed data, RMSE climbs (use to force retrain in demos)

Toggle in the UI — enable whichever suits the demo narrative.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.monitoring.simulator import run_one_batch


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "retries": 0,
    "retry_delay": timedelta(minutes=1),
    "email_on_failure": False,
}


with DAG(
    dag_id="simulator_normal",
    description="Replay holdout as live predictions + ground truth (clean)",
    default_args=default_args,
    schedule=timedelta(minutes=5),
    catchup=False,
    max_active_runs=1,
    tags=["simulator", "aqi"],
) as dag_normal:
    PythonOperator(
        task_id="replay_clean_batch",
        python_callable=lambda: run_one_batch(inject_drift=False),
    )


with DAG(
    dag_id="simulator_drift",
    description="Replay holdout with injected drift (forces retrain)",
    default_args=default_args,
    schedule=None,  # Manual trigger only — don't run by default
    catchup=False,
    max_active_runs=1,
    tags=["simulator", "aqi", "demo"],
) as dag_drift:
    PythonOperator(
        task_id="replay_drift_batch",
        python_callable=lambda: run_one_batch(inject_drift=True),
    )