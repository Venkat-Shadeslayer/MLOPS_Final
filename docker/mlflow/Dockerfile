# ============================================================================
# MLflow tracking server + model registry
# ============================================================================
FROM python:3.11-slim

WORKDIR /mlflow

RUN pip install --no-cache-dir \
        mlflow==2.13.0 \
        psycopg2-binary==2.9.9 \
        boto3==1.34.106

EXPOSE 5000

CMD mlflow server \
    --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
    --default-artifact-root ${MLFLOW_ARTIFACT_ROOT} \
    --host 0.0.0.0 \
    --port 5000