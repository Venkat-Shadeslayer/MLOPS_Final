"""
Data pipeline DAG: ingest -> validate -> feature_engineer.

Schedule: daily at 02:00. In demo, can be triggered manually from Airflow UI.
This is the operational counterpart to the DVC pipeline; DVC handles
reproducibility, Airflow handles scheduling and operational logs.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Importable because docker-compose mounts ./src into /opt/airflow/src
# and Dockerfile sets PYTHONPATH=/opt/airflow:/opt/airflow/src
from src.data.ingest import ingest
from src.data.validate import main as validate_main
from src.features.transform import main as transform_main


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}


with DAG(
    dag_id="data_pipeline",
    description="Ingest -> validate -> feature engineer the AQI dataset",
    default_args=default_args,
    schedule="0 2 * * *",
    catchup=False,
    max_active_runs=1,
    tags=["data", "aqi"],
) as dag:

    ingest_task = PythonOperator(
        task_id="ingest_raw_csv",
        python_callable=ingest,
    )

    validate_task = PythonOperator(
        task_id="validate_schema_and_quality",
        python_callable=validate_main,
    )

    feature_task = PythonOperator(
        task_id="engineer_features_and_baseline",
        python_callable=transform_main,
    )

    ingest_task >> validate_task >> feature_task