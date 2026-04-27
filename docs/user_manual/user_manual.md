# User Manual — AQI MLOps Console

*For non-technical users: pollution analysts, city planners, evaluators.*

## What this application does

Predicts the **Air Quality Index (AQI)** for an Indian city-day from 12 pollutant
measurements. The system continuously learns from real observations you submit
as feedback, so the model gets better over time.

## Getting started (one time)

1. Ask your admin for the URL. If running locally, it's **http://localhost:8501**.
2. You'll see the home page with:
   - **API status** — green means the prediction engine is online.
   - **Model version** — the version of the AQI model currently deployed.
   - **Rolling RMSE** — how accurate the model has been in the last 24 hours (lower is better).

No login is needed for the prediction UI.

## The four pages

The sidebar on the left has four navigation links.

### 🌫️ Home
Landing page — a quick system status. If any tile is red, contact your admin.

### 📈 Predict
Get a prediction for a specific city-day.

1. Select a city from the dropdown.
2. Pick the date.
3. Enter pollutant concentrations (µg/m³ for most, mg/m³ for CO). If you don't know a value, leave it at 0 — the model handles missing values gracefully.
4. Click **Predict AQI**.
5. The result shows:
   - Predicted AQI value
   - CPCB category (Good / Satisfactory / Moderate / Poor / Very Poor / Severe) with a color band
   - The prediction ID — **copy this if you want to submit feedback later.**

### 🎯 Feedback
Tell the system what the actual AQI was — this is how the model learns.

**What you see:**

1. **Retrain-trigger panel (top)** — shows how many feedbacks have been collected in the window, current rolling RMSE, and whether each gate is met. A progress bar shows how close you are to triggering retraining. By default: retraining fires when **≥ 10 feedbacks** have been collected in 24h **AND** rolling RMSE exceeds **100**. Both numbers are configurable in `.env` — see the "How to change the thresholds" expander on the page.
2. **Submit section (middle)** — two ways to submit:
   - **From this session:** predictions you made on the Predict page appear as cards — fill in the actual AQI and click Submit.
   - **By prediction ID:** paste a prediction ID from an earlier session and submit the actual value.
3. **Feedback history (bottom)** — a table of all past feedback with predicted vs actual, absolute error, model version. Use the ⬇️ **Download CSV** button to export everything for audit or external analysis.

Each submission refreshes the rolling RMSE and progress bar in real time.

### 🔧 Pipeline
Operator/engineer view. Shows:

- A flowchart of the end-to-end ML pipeline
- Live tabs embedding:
  - **Airflow** — see which pipeline jobs are running or have failed
  - **MLflow** — browse experiments and model versions
  - **Grafana** — real-time dashboards (latency, prediction rate, drift)

## Understanding the AQI categories (CPCB India)

| AQI range | Category | Color |
|---|---|---|
| 0–50 | Good | 🟢 |
| 51–100 | Satisfactory | 🟢 |
| 101–200 | Moderate | 🟡 |
| 201–300 | Poor | 🟠 |
| 301–400 | Very Poor | 🔴 |
| 401+ | Severe | 🟣 |

## FAQ

**Q: I submitted a prediction but the Home page still shows "no rolling RMSE".**
A: RMSE is computed only on predictions that have received ground-truth feedback. Submit an actual value via the Feedback page.

**Q: I made a mistake entering pollutant values.**
A: Just run another prediction with the corrected values. Old predictions are kept for audit but the new one will be the latest.

**Q: The model version jumped from v5 to v6 — what happened?**
A: The system auto-retrained. This happens either (a) on a scheduled cadence, (b) when the recent accuracy (rolling RMSE) falls below an acceptable threshold, or (c) when operators manually trigger `training_pipeline` in Airflow. The new model is automatically deployed.

**Q: I see a 🔴 API tile on the Home page.**
A: The prediction service is down. Contact your admin. Nothing else on the UI will work until this is resolved.

**Q: Can I trust the prediction?**
A: The model's validation RMSE is visible in MLflow (Pipeline page → MLflow tab → pick a run). If the rolling RMSE is much higher than validation RMSE, the model has drifted — an automatic retrain will kick in.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Page shows "Model not ready" | No Production model registered yet | Trigger `training_pipeline` in Airflow (http://localhost:8080) |
| Pipeline page iframes empty | Grafana/Airflow containers down | `docker compose ps` → restart failed services |
| "Cannot reach API" error | API container crashed or restarting | `docker compose logs api` |
| Predictions look wildly wrong | Model stale / concept drift | Submit 3–5 ground-truth feedbacks with real values; drift monitor will trigger retrain |

## Admin links (for operators only)

| Tool | URL | Purpose |
|---|---|---|
| Airflow | http://localhost:8080 | Trigger/inspect DAGs |
| MLflow | http://localhost:5001 | Experiments + registry |
| Grafana | http://localhost:3001 | Dashboards + alerts |
| Prometheus | http://localhost:9090 | Raw metrics |
| API docs | http://localhost:8000/docs | OpenAPI Swagger |

Default Airflow/Grafana credentials are in `.env`.
