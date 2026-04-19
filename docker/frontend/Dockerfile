# ============================================================================
# Streamlit frontend
# ============================================================================
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY docker/frontend/requirements.txt /tmp/frontend-requirements.txt
RUN pip install --no-cache-dir -r /tmp/frontend-requirements.txt

COPY frontend/ /app/frontend/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "/app/frontend/Home.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]