"""
Drift monitor DAG.

Every 10 minutes:
  1. compute_drift — PSI per feature, pushed to Prometheus
  2. check_decay   — combines PSI + rolling RMSE
  3. branch        — if decay detected, trigger training_pipeline

When retrain triggers, training_pipeline runs end-to-end and produces a new
Production model version automatically.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from src.monitoring.decay_check import check_decay
from src.monitoring.drift import main as drift_main


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "retries": 0,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}


def _decay_or_skip(**context) -> str:
    """Branch function: returns next task_id."""
    decision = check_decay()
    context["ti"].xcom_push(key="decay_decision", value=decision)
    return "trigger_retrain" if decision["should_retrain"] else "no_retrain_needed"


def _no_retrain():
    # Explicit no-op so Airflow shows a clear green terminator
    print("No retrain needed.")


with DAG(
    dag_id="drift_monitor",
    description="Compute drift, check decay, trigger retraining if needed",
    default_args=default_args,
    schedule=timedelta(minutes=10),
    catchup=False,
    max_active_runs=1,
    tags=["monitoring", "aqi"],
) as dag:

    compute_drift = PythonOperator(
        task_id="compute_drift",
        python_callable=drift_main,
    )

    branch = BranchPythonOperator(
        task_id="check_decay_and_branch",
        python_callable=_decay_or_skip,
    )

    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="training_pipeline",
        wait_for_completion=False,
        reset_dag_run=True,
    )

    no_retrain = PythonOperator(
        task_id="no_retrain_needed",
        python_callable=_no_retrain,
    )

    compute_drift >> branch >> [trigger_retrain, no_retrain]