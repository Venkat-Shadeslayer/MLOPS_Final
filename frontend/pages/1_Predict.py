"""
Prediction page.

Non-technical UX: user picks a city, enters 12 pollutants, hits predict.
We store the prediction in session state so Feedback page can surface it.
Inputs have tooltips and sensible defaults (India urban averages).
"""
from __future__ import annotations

from datetime import date

import streamlit as st

from frontend.api_client import APIError, aqi_bucket, predict

st.set_page_config(page_title="Predict — AQI MLOps", page_icon="📈", layout="wide")
st.title("📈 AQI prediction")
st.caption("Enter today's pollutant readings for a city. Defaults shown are typical Indian urban values.")

# ---- Cities (from training data) -------------------------------------------
CITIES = [
    "Ahmedabad", "Aizawl", "Amaravati", "Amritsar", "Bengaluru", "Bhopal",
    "Brajrajnagar", "Chandigarh", "Chennai", "Coimbatore", "Delhi", "Ernakulam",
    "Gurugram", "Guwahati", "Hyderabad", "Jaipur", "Jorapokhar", "Kochi",
    "Kolkata", "Lucknow", "Mumbai", "Patna", "Shillong", "Talcher",
    "Thiruvananthapuram", "Visakhapatnam",
]

# ---- Initialize session state ----------------------------------------------
if "recent_predictions" not in st.session_state:
    st.session_state.recent_predictions = []

# ---- Input form ------------------------------------------------------------
with st.form("predict_form", clear_on_submit=False):
    c1, c2, c3 = st.columns([2, 2, 1])
    city = c1.selectbox("City", CITIES, index=CITIES.index("Delhi"))
    when = c2.date_input("Date", value=date.today())
    _ = c3.empty()

    st.markdown("**Pollutants (µg/m³ unless noted)**")

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    pm25 = r1c1.number_input("PM2.5", min_value=0.0, max_value=2000.0, value=80.0, step=5.0,
                              help="Fine particulates. WHO guideline: <15 µg/m³ annual.")
    pm10 = r1c2.number_input("PM10", min_value=0.0, max_value=3000.0, value=140.0, step=5.0,
                              help="Coarse particulates. WHO guideline: <45 µg/m³ annual.")
    no_val = r1c3.number_input("NO", min_value=0.0, max_value=500.0, value=20.0, step=1.0)
    no2 = r1c4.number_input("NO₂", min_value=0.0, max_value=500.0, value=35.0, step=1.0,
                              help="Nitrogen dioxide.")

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    nox = r2c1.number_input("NOₓ", min_value=0.0, max_value=1000.0, value=45.0, step=1.0)
    nh3 = r2c2.number_input("NH₃", min_value=0.0, max_value=500.0, value=15.0, step=1.0,
                              help="Ammonia.")
    co = r2c3.number_input("CO (mg/m³)", min_value=0.0, max_value=100.0, value=1.2, step=0.1,
                              help="Carbon monoxide — note mg/m³ unit.")
    so2 = r2c4.number_input("SO₂", min_value=0.0, max_value=500.0, value=10.0, step=1.0,
                              help="Sulfur dioxide.")

    r3c1, r3c2, r3c3, r3c4 = st.columns(4)
    o3 = r3c1.number_input("O₃", min_value=0.0, max_value=500.0, value=40.0, step=1.0,
                              help="Ozone.")
    benzene = r3c2.number_input("Benzene", min_value=0.0, max_value=200.0, value=2.0, step=0.1)
    toluene = r3c3.number_input("Toluene", min_value=0.0, max_value=200.0, value=4.0, step=0.1)
    xylene = r3c4.number_input("Xylene", min_value=0.0, max_value=200.0, value=1.0, step=0.1)

    submitted = st.form_submit_button("🔮 Predict AQI", type="primary", use_container_width=True)

# ---- Submit ----------------------------------------------------------------
if submitted:
    reading = {
        "city": city,
        "date": str(when),
        "PM2.5": pm25, "PM10": pm10, "NO": no_val, "NO2": no2,
        "NOx": nox, "NH3": nh3, "CO": co, "SO2": so2, "O3": o3,
        "Benzene": benzene, "Toluene": toluene, "Xylene": xylene,
    }
    try:
        with st.spinner("Predicting..."):
            result = predict(reading)
    except APIError as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Show result
    pred_value = result["predicted_aqi"]
    label, color = aqi_bucket(pred_value)

    st.markdown("---")
    big_col, info_col = st.columns([2, 1])
    with big_col:
        st.markdown(
            f"""
            <div style="padding:24px;border-radius:12px;background-color:{color};
                        text-align:center;color:white;">
              <div style="font-size:14px;opacity:0.9;">Predicted AQI for {city}</div>
              <div style="font-size:72px;font-weight:700;line-height:1;">{pred_value:.1f}</div>
              <div style="font-size:24px;font-weight:500;margin-top:8px;">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with info_col:
        st.metric("Model version", f"v{result['model_version']}")
        st.metric("Latency", f"{result['latency_ms']:.1f} ms")
        st.caption(f"Prediction ID: `{result['prediction_id']}`")
        st.caption("Copy this ID if you want to submit the actual AQI later on the Feedback page.")

    # Remember it
    st.session_state.recent_predictions.insert(0, {
        "prediction_id": result["prediction_id"],
        "city": city,
        "date": str(when),
        "predicted_aqi": pred_value,
        "model_version": result["model_version"],
    })
    st.session_state.recent_predictions = st.session_state.recent_predictions[:10]

# ---- Recent predictions list -----------------------------------------------
if st.session_state.recent_predictions:
    st.markdown("---")
    st.subheader("Recent predictions (this session)")
    for rec in st.session_state.recent_predictions:
        label, color = aqi_bucket(rec["predicted_aqi"])
        st.markdown(
            f"- **{rec['city']}** on `{rec['date']}` → **{rec['predicted_aqi']:.1f}** "
            f"({label}) · model v{rec['model_version']} · id `{rec['prediction_id'][:8]}…`"
        )