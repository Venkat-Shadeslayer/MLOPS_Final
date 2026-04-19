"""Streamlit landing page. Day 2 stub — full UI ships Day 7."""
import os
import streamlit as st

st.set_page_config(page_title="AQI MLOps", page_icon="🌫️", layout="wide")

st.title("AQI MLOps Console")
st.markdown(
    "End-to-end MLOps system for Air Quality Index prediction. "
    "Full prediction UI and pipeline visualization ship in upcoming days."
)

st.subheader("System status")
api_url = os.getenv("STREAMLIT_API_URL", "http://api:8000")
st.write(f"**API endpoint:** `{api_url}`")

st.info("Day 2 placeholder. Real UI coming soon.")

with st.expander("Useful links"):
    st.markdown(
        "- **Airflow**: http://localhost:8080\n"
        "- **MLflow**: http://localhost:5001\n"
        "- **Grafana**: http://localhost:3001\n"
        "- **Prometheus**: http://localhost:9090\n"
        "- **API docs**: http://localhost:8000/docs"
    )