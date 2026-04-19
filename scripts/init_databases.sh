#!/bin/bash
# ============================================================================
# Postgres bootstrap — creates the three project databases on first boot.
# Runs exactly once via /docker-entrypoint-initdb.d hook.
# ============================================================================
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE airflow;
    CREATE DATABASE mlflow;
    CREATE DATABASE predictions;
    GRANT ALL PRIVILEGES ON DATABASE airflow TO $POSTGRES_USER;
    GRANT ALL PRIVILEGES ON DATABASE mlflow TO $POSTGRES_USER;
    GRANT ALL PRIVILEGES ON DATABASE predictions TO $POSTGRES_USER;
EOSQL

echo "Created databases: airflow, mlflow, predictions"