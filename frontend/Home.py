"""
AQI MLOps — landing page.

First thing a user sees. Shows system health, current model, quick stats.
Acts as the map: directs users to Predict / Feedback / Pipeline pages via
the sidebar.
"""
from __future__ import annotations

import streamlit as st

from frontend.api_client import APIError, health, ready, stats

st.set_page_config(
    page_title="AQI MLOps Console",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌫️ AQI MLOps Console")
st.caption("Air Quality Index prediction with full MLOps lifecycle — IIT Madras, sem 8")

# ---- System status ----------------------------------------------------------
st.subheader("System status")
col1, col2, col3 = st.columns(3)

try:
    h = health()
    col1.metric("API", "🟢 online", help=f"Service: {h.get('service')}")
except APIError as e:
    col1.metric("API", "🔴 offline")
    st.error(f"API unreachable: {e}")
    st.stop()

try:
    r = ready()
    if r.get("ready"):
        col2.metric(
            "Model",
            f"v{r.get('model_version')}",
            help=f"Production model: {r.get('model_name')}",
        )
    else:
        col2.metric("Model", "not ready")
        st.warning(f"Model not loaded: {r.get('detail')}")
except APIError as e:
    col2.metric("Model", "error")
    st.error(str(e))

try:
    s = stats()
    rmse = s.get("rolling_rmse_24h")
    col3.metric(
        "Rolling RMSE (24h)",
        f"{rmse:.2f}" if rmse is not None else "—",
        help="Rolling 24-hour RMSE from predictions with ground-truth feedback",
    )
except APIError:
    s = {}
    col3.metric("Rolling RMSE (24h)", "—")

# ---- Retrain trigger rule ---------------------------------------------------
if s:
    ct = s.get("feedback_count_window", 0) or 0
    cth = s.get("feedback_count_threshold", 10)
    rth = s.get("rmse_threshold", 100.0)
    win = s.get("window_hours", 24)
    banner = (
        f"🔁 **Auto-retrain rule:** {ct} / {cth} feedbacks collected in the last {win}h. "
        f"Retraining fires when ≥{cth} feedbacks AND rolling RMSE > {rth:.0f}. "
        f"Adjust on the Feedback page."
    )
    if s.get("count_gate_met") and s.get("rmse_gate_met"):
        st.error(banner + "  \n⚡ Both gates met — next drift check will trigger retraining.")
    else:
        st.info(banner)

# ---- What this does ---------------------------------------------------------
st.subheader("What this does")
st.markdown(
    "**Predicts Air Quality Index (AQI)** from 12 pollutant measurements: "
    "PM2.5, PM10, NO, NO₂, NOₓ, NH₃, CO, SO₂, O₃, Benzene, Toluene, Xylene.\n\n"
    "The model is an **XGBoost regressor** trained on 5+ years of Indian city-day "
    "observations. Feature engineering adds lag features (t-1, t-2, t-3, t-7 days) "
    "and rolling means (3, 7, 14 days) per city. Predictions are continuously "
    "monitored — if model accuracy degrades on real ground truth, an Airflow DAG "
    "automatically retrains and registers a new version."
)

# ---- Navigation hint --------------------------------------------------------
st.subheader("Where to go")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("### [📈 Predict](/Predict)")
    st.markdown(
        "Enter pollutant readings for a city and get an AQI prediction."
    )
with c2:
    st.markdown("### [🎯 Feedback](/Feedback)")
    st.markdown(
        "Submit the actual observed AQI for a past prediction.  \n"
        "This drives the closed-loop retraining."
    )
with c3:
    st.markdown("### [🔧 Pipeline](/Pipeline)")
    st.markdown(
        "Visualize the ML pipeline — data flow, training runs, live metrics, "
        "drift, retraining events."
    )

with st.expander("Operator links"):
    st.markdown(
        "- **Airflow** (DAG orchestration): http://localhost:8080\n"
        "- **MLflow** (experiments + registry): http://localhost:5001\n"
        "- **Grafana** (dashboards): http://localhost:3001\n"
        "- **Prometheus** (metrics): http://localhost:9090\n"
        "- **API docs** (OpenAPI): http://localhost:8000/docs"
    )