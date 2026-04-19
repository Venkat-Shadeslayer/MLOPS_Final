"""
Training DAG: train XGBoost + NN in parallel, then register the winner.

Schedule: manual trigger (schedule=None). Day 5 adds the drift-triggered
retrain that fires this DAG automatically.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.models.nn_trainer import train as train_nn
from src.models.register import register_best
from src.models.xgboost_trainer import train as train_xgb


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "retries": 0,  # don't silently retry failed model trainings — surface the error
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}


with DAG(
    dag_id="training_pipeline",
    description="Train XGBoost + NN, compare, register winner to Production",
    default_args=default_args,
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["training", "aqi"],
) as dag:

    train_xgb_task = PythonOperator(
        task_id="train_xgboost",
        python_callable=train_xgb,
    )

    train_nn_task = PythonOperator(
        task_id="train_pytorch_nn",
        python_callable=train_nn,
    )

    register_task = PythonOperator(
        task_id="register_best_model",
        python_callable=register_best,
    )

    # XGB and NN run in parallel, register waits for both
    [train_xgb_task, train_nn_task] >> register_task