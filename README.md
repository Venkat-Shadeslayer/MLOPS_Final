# AQI MLOps — Air Quality Index Prediction with Full MLOps Lifecycle

End-to-end MLOps application predicting Air Quality Index (AQI) from 12 pollutant
measurements, with closed-loop retraining driven by user feedback and data drift.

## Quick start

```bash
cp .env.example .env
docker compose up -d
```

Service URLs:

| Service    | URL                         |
|------------|-----------------------------|
| Frontend   | http://localhost:8501       |
| API        | http://localhost:8000/docs  |
| Airflow    | http://localhost:8080       |
| MLflow     | http://localhost:5001       |
| Grafana    | http://localhost:3001       |
| Prometheus | http://localhost:9090       |

## Stack

- **Model:** XGBoost regressor + PyTorch NN (winner auto-registered)
- **Orchestration:** Airflow (data pipeline + training + drift monitor)
- **Tracking & registry:** MLflow
- **Serving:** FastAPI (Postgres-backed predictions log)
- **Frontend:** Streamlit multipage (Home / Predict / Feedback / Pipeline)
- **Monitoring:** Prometheus + Grafana, PSI-based drift, rolling-RMSE decay
- **Versioning:** Git + DVC
- **Packaging:** Docker Compose (8 services)

## Documentation

See [docs/](docs/) for:

- [Architecture diagram](docs/architecture/architecture.md)
- [High-Level Design](docs/hld/HLD.md)
- [Low-Level Design + API spec](docs/lld/LLD.md)
- [Test plan & report](docs/test_plan/test_plan.md)
- [User manual](docs/user_manual/user_manual.md)
