"""
Isolated SMTP-failure test DAG.

Triggers an intentional task failure to verify Airflow's email_on_failure
hook reaches Mailtrap. Demo-only: paused by default once verified.

Filename starts with `_` so it sorts above the real DAGs in the UI.
"""
from __future__ import annotations

import os
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def _fail_loudly() -> None:
    raise RuntimeError("Intentional failure to test SMTP/Mailtrap wiring.")


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "retries": 0,
    "email_on_failure": True,
    "email": [os.getenv("ALERT_EMAIL_TO", "admin@aqi-mlops.local")],
}


with DAG(
    dag_id="_smtp_test",
    description="Verify SMTP wiring end-to-end. Fails on purpose; email_on_failure=True.",
    default_args=default_args,
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["meta", "smoke"],
) as dag:

    PythonOperator(
        task_id="raise_exception",
        python_callable=_fail_loudly,
    )
