# User Manual — AQI MLOps Console

This manual is for the **end user** of the AQI MLOps web application. It explains
how to start the system, generate predictions, submit feedback, and inspect the
pipeline. No knowledge of the backend services is required.

---

## 1. What the application does

The AQI MLOps Console predicts the **Air Quality Index (AQI)** for a city from
12 pollutant readings (PM2.5, PM10, NO, NO₂, NOₓ, NH₃, CO, SO₂, O₃, Benzene,
Toluene, Xylene). Every prediction is logged. When you submit the **actual**
observed AQI as feedback, the system measures its own error and automatically
retrains the model when error or drift cross configured thresholds.

---

## 2. Prerequisites

- macOS, Linux, or Windows with **Docker Desktop** running.
- 8 GB of free RAM and ~5 GB of free disk space.
- Network access on first launch (for image and dataset download).

You do **not** need Python, Airflow, or MLflow installed locally — everything
runs inside Docker containers.

---

## 3. Starting the system

From the project directory:

```bash
cp .env.example .env
docker compose up -d --build
```

Wait roughly two minutes for all services to become healthy. Then bootstrap the
first model (one-time setup):

```bash
docker compose exec airflow airflow dags unpause data_pipeline
docker compose exec airflow airflow dags trigger data_pipeline
# wait ~2 min
docker compose exec airflow airflow dags unpause training_pipeline
docker compose exec airflow airflow dags trigger training_pipeline
# wait ~5 min
docker compose restart api
```

When `curl http://localhost:8000/ready` reports `"ready": true`, the system is
ready to use.

---

## 4. Opening the web app

Open **http://localhost:8501** in any modern browser. The console has four
pages, accessible from the left sidebar:

| Page       | Purpose                                             |
|------------|-----------------------------------------------------|
| Home       | System status, deployed model version, retrain gate |
| Predict    | Enter pollutant readings → get an AQI prediction    |
| Feedback   | Submit the actual AQI for a past prediction         |
| Pipeline   | Live pipeline view, latency self-test, run history  |

---

## 5. Generating a prediction (Predict page)

1. Select a **city** and **date**.
2. Enter the pollutant values you have. Fields you leave blank are treated as
   zero — supply at least PM2.5, PM10, NO₂, and CO for a meaningful result.
3. Click **Predict AQI**.
4. The page displays:
   - the predicted AQI value and its category (Good / Moderate / Poor / …),
   - the model version that produced it (e.g. `v14`),
   - the request latency in milliseconds,
   - a **prediction ID** — copy this if you intend to submit ground truth later.

---

## 6. Submitting feedback (Feedback page)

When you later observe the **actual** AQI for a prediction:

1. Open the **Feedback** page.
2. Paste the **prediction ID** from the Predict result.
3. Enter the observed AQI value.
4. Click **Submit feedback**.

The page shows the running state of the two retraining gates:

- **Count gate** — number of feedback records in the active 24-hour window
  versus the configured threshold (default `4` for demo, `10` for production).
- **RMSE gate** — rolling RMSE of the feedback window versus the configured
  threshold.

When both gates are met, a banner indicates the next drift-monitor run will
trigger an automatic retrain. You can also browse the full feedback history
and download it as CSV from the same page.

---

## 7. Inspecting the pipeline (Pipeline page)

The Pipeline page is a single operations view that consolidates:

- a Sankey-style flow of ingestion → features → training → inference,
- a **latency self-test** (p50, p95, mean, throughput),
- the most recent Airflow DAG runs,
- embedded panels for MLflow, Airflow, and Grafana.

Use this page to confirm the system is healthy and to follow a retraining run
end-to-end without leaving the browser.

---

## 8. Troubleshooting

| Symptom                                    | Likely cause / fix                                                                  |
|--------------------------------------------|-------------------------------------------------------------------------------------|
| Home page shows `API: down`                | API container restarting — wait 30 seconds, then refresh.                           |
| `/ready` returns `model_loaded: false`     | No Production model yet — run the one-time bootstrap in §3.                         |
| Predict button greyed out                  | Required pollutant fields missing or out of valid range.                            |
| `prediction_id not found` on feedback      | The ID was mistyped, or the predictions DB was reset.                               |
| Retraining gates never trip in demo        | Lower `FEEDBACK_COUNT_THRESHOLD` and `DRIFT_RMSE_THRESHOLD` in `.env`, then restart. |
| Stop the system                            | `docker compose down` (preserves data) or `docker compose down -v` (wipes data).    |

For deeper diagnostics, the API log is available at
`docker compose logs -f api`, and the Airflow UI at **http://localhost:8080**
shows individual task logs.

---

## 9. Where to go next

- **Service URLs and credentials:** see the table in the project [README.md](../README.md).
- **API contract:** [docs/lld/LLD.md](lld/LLD.md) or the live OpenAPI page at
  http://localhost:8000/docs.
- **Architecture and design rationale:** [docs/hld/HLD.md](hld/HLD.md).
