# ============================================================================
# Airflow image — official base + project ML deps
# ----------------------------------------------------------------------------
# Uses Airflow's published constraints file so our extra deps (mlflow, torch,
# xgboost, etc.) install at versions known compatible with Airflow 2.9.1.
# Without --constraint, pip's resolver fights Airflow's pinned core deps
# (sqlalchemy, pandas, flask) and emits ResolutionImpossible.
# ============================================================================
FROM apache/airflow:2.9.1-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

USER airflow

ARG AIRFLOW_VERSION=2.9.1
ARG PYTHON_VERSION=3.11
ARG CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

COPY docker/airflow/requirements.txt /tmp/airflow-requirements.txt
RUN pip install --no-cache-dir \
        --constraint "${CONSTRAINT_URL}" \
        -r /tmp/airflow-requirements.txt

ENV PYTHONPATH=/opt/airflow:/opt/airflow/src:${PYTHONPATH}