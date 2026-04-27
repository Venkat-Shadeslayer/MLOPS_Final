"""
Feedback page — closes the MLOps loop.

Three sections:
  1. Retrain-trigger status panel (count-gate + RMSE-gate, thresholds from API)
  2. Submit ground truth (quick-fill from session + manual by prediction ID)
  3. Feedback history table + CSV download
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.api_client import (
    APIError,
    feedback_csv_bytes,
    feedback_list,
    stats,
    submit_ground_truth,
)

st.set_page_config(page_title="Feedback — AQI MLOps", page_icon="🎯", layout="wide")
st.title("🎯 Feedback & closed-loop retraining")
st.caption(
    "Report actual observed AQI for past predictions. "
    "Enough high-error feedback triggers automatic retraining."
)

# ---------------------------------------------------------------------------
# 1. Retrain-trigger status panel
# ---------------------------------------------------------------------------
st.subheader("Retraining trigger")

try:
    s = stats()
except APIError as e:
    st.error(f"Could not fetch stats: {e}")
    s = {}

count = s.get("feedback_count_window", 0) or 0
count_thresh = s.get("feedback_count_threshold", 10)
rmse = s.get("rolling_rmse_24h")
rmse_thresh = s.get("rmse_threshold", 100.0)
window = s.get("window_hours", 24)
count_gate = s.get("count_gate_met", False)
rmse_gate = s.get("rmse_gate_met", False)
will_retrain = count_gate and rmse_gate

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    f"Feedbacks in last {window}h",
    f"{count} / {count_thresh}",
    help="Retraining needs at least this many feedback rows in the window.",
)
c2.metric(
    "Rolling RMSE",
    f"{rmse:.2f}" if rmse is not None else "—",
    help=f"Threshold: {rmse_thresh:.1f}. Retrain fires when RMSE exceeds this.",
)
c3.metric(
    "Count gate",
    "✅ met" if count_gate else "⏳ pending",
)
c4.metric(
    "RMSE gate",
    "🔴 breach" if rmse_gate else "🟢 ok",
)

# Progress bar for the count gate
progress = min(count / max(count_thresh, 1), 1.0)
st.progress(progress, text=f"Feedback progress: {count} of {count_thresh}")

# Clear plain-English rule statement
rule = (
    f"**Trigger rule:** retrain when **≥ {count_thresh} feedbacks** have been "
    f"received in the last {window}h **AND** rolling RMSE > **{rmse_thresh:.1f}**. "
    f"A separate PSI-drift gate can also trigger retraining on input distribution shift."
)
if will_retrain:
    st.error(rule + "\n\n⚡ **Both gates are currently met — the next drift_monitor run (every 10 min) will trigger retraining.**")
else:
    st.info(rule)

with st.expander("How to change the thresholds"):
    st.markdown(
        "Edit `.env` and restart the API:\n"
        "```\n"
        f"FEEDBACK_COUNT_THRESHOLD={count_thresh}   # how many feedbacks required\n"
        f"DRIFT_RMSE_THRESHOLD={rmse_thresh}        # RMSE above which retrain fires\n"
        f"DRIFT_CHECK_WINDOW_HOURS={window}         # window size\n"
        "```\n"
        "Then: `docker compose restart api airflow` to pick up changes."
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# 2. Submit ground truth
# ---------------------------------------------------------------------------
st.subheader("Submit ground truth")

recent = st.session_state.get("recent_predictions", [])
if recent:
    st.caption("Quick-fill from predictions made in this session:")
    for rec in recent:
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 2, 2])
            c1.markdown(
                f"**{rec['city']}** on `{rec['date']}`  \n"
                f"Predicted: **{rec['predicted_aqi']:.1f}** · model v{rec['model_version']}"
            )
            actual = c2.number_input(
                "Actual AQI",
                min_value=0.0,
                max_value=1000.0,
                value=float(rec["predicted_aqi"]),
                step=1.0,
                key=f"actual_{rec['prediction_id']}",
            )
            if c3.button("Submit", key=f"submit_{rec['prediction_id']}"):
                try:
                    resp = submit_ground_truth(rec["prediction_id"], actual)
                    st.success(
                        f"Recorded. Rolling RMSE (24h): {resp.get('rolling_rmse_24h', '—')}"
                    )
                    st.rerun()
                except APIError as e:
                    st.error(f"Failed: {e}")

with st.form("manual_submit"):
    st.caption("…or submit for any past prediction by ID:")
    c1, c2 = st.columns([3, 1])
    pred_id = c1.text_input("Prediction ID", placeholder="UUID from a past /predict response")
    actual = c2.number_input("Actual AQI", min_value=0.0, max_value=1000.0, value=100.0, step=1.0)
    go = st.form_submit_button("Submit ground truth", type="primary")

if go:
    if not pred_id.strip():
        st.warning("Paste a prediction ID first.")
    else:
        try:
            resp = submit_ground_truth(pred_id.strip(), actual)
            st.success(f"Recorded. Rolling RMSE (24h): {resp.get('rolling_rmse_24h', '—')}")
            st.rerun()
        except APIError as e:
            st.error(f"Failed: {e}")

st.markdown("---")

# ---------------------------------------------------------------------------
# 3. Feedback history + CSV download
# ---------------------------------------------------------------------------
st.subheader("Feedback history")

try:
    fb = feedback_list(limit=500)
    rows = fb.get("rows", [])
except APIError as e:
    st.warning(f"Could not load feedback: {e}")
    rows = []

if not rows:
    st.info("No feedback submitted yet.")
else:
    df = pd.DataFrame(rows)
    # Sensible column order for display
    display_cols = [
        "feedback_at", "city", "date",
        "predicted_aqi", "actual_aqi", "abs_error",
        "model_version", "model_family", "prediction_id",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "predicted_aqi": st.column_config.NumberColumn(format="%.1f"),
            "actual_aqi": st.column_config.NumberColumn(format="%.1f"),
            "abs_error": st.column_config.NumberColumn("|error|", format="%.1f"),
        },
    )

    c1, c2 = st.columns([1, 4])
    try:
        csv_bytes = feedback_csv_bytes()
        c1.download_button(
            label="⬇️ Download CSV",
            data=csv_bytes,
            file_name="feedback.csv",
            mime="text/csv",
            type="primary",
        )
    except APIError as e:
        c1.warning(f"CSV unavailable: {e}")
    c2.caption(f"Showing {len(rows)} rows. CSV download fetches up to 10 000 rows.")
