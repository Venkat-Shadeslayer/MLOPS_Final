# ============================================================================
# FastAPI inference service
# Loads latest Production model from MLflow registry on startup.
# Exposes /predict /health /ready /metrics
# ============================================================================
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY docker/api/requirements.txt /tmp/api-requirements.txt
RUN pip install --no-cache-dir -r /tmp/api-requirements.txt

COPY src/ /app/src/

ENV PYTHONPATH=/app:/app/src

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]