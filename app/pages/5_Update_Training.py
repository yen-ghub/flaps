"""
Update data and retrain models from within the Streamlit app.
"""

import io
import sys
import os
import contextlib
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
from src.ui_theme import apply_theme

st.set_page_config(page_title="FLAPS — Update & Train", page_icon="✈️", layout="wide")
apply_theme()
st.title("Update & Train")
st.markdown(
    "Download the latest data from BOM and BITRE, then retrain models. "
    "Each operation may take several minutes."
)


class StreamlitLogger(io.StringIO):
    """Captures print output and writes it to a Streamlit container in real time."""

    def __init__(self, container):
        super().__init__()
        self._container = container
        self._lines = []

    def write(self, text):
        if text and text.strip():
            self._lines.append(text.rstrip())
            self._container.code("\n".join(self._lines), language="text")
        return len(text)

    def flush(self):
        pass


st.divider()

# ---- 1. Update Data ----
st.subheader("1. Update Data")
st.markdown(
    "Downloads the latest weather observations from BOM FTP and flight performance "
    "data from BITRE, then merges them into the training dataset."
)

if st.button("Update Data", type="primary", key="btn_update"):
    from src.data_loader import update_all_data

    log_container = st.empty()
    logger = StreamlitLogger(log_container)

    with st.status("Updating data...", expanded=True) as status:
        old_stdout = sys.stdout
        sys.stdout = logger
        try:
            update_all_data()
            sys.stdout = old_stdout
            status.update(label="Data update complete.", state="complete", expanded=False)
            st.success("Data updated successfully. You can now retrain the models below.")
        except Exception:
            sys.stdout = old_stdout
            status.update(label="Data update failed.", state="error", expanded=True)
            st.error("Data update failed. See log above for details.")
            st.code(traceback.format_exc(), language="text")

    # Clear cached data so the app picks up the new dataset
    st.cache_data.clear()

st.divider()

# ---- 2. Retrain NOWCASTING Models ----
st.subheader("2. Retrain NOWCASTING Models")
st.markdown(
    "Retrains NOWCASTING models (Ridge, Random Forest, Logistic, XGBoost, Neural Network) "
    "using same-month weather features. Saves to `models/nowcasting/`."
)

if st.button("Retrain NOWCASTING Models", type="primary", key="btn_retrain"):
    from src.train_and_save import train_and_save

    log_container = st.empty()
    logger = StreamlitLogger(log_container)

    with st.status("Training NOWCASTING models...", expanded=True) as status:
        old_stdout = sys.stdout
        sys.stdout = logger
        try:
            train_and_save()
            sys.stdout = old_stdout
            status.update(label="NOWCASTING training complete.", state="complete", expanded=False)
            st.success("NOWCASTING models retrained and saved successfully.")
        except Exception:
            sys.stdout = old_stdout
            status.update(label="NOWCASTING training failed.", state="error", expanded=True)
            st.error("Training failed. See log above for details.")
            st.code(traceback.format_exc(), language="text")

    st.cache_data.clear()
    st.cache_resource.clear()

st.divider()

# ---- 3. Retrain FORECASTING Models ----
st.subheader("3. Retrain FORECASTING Models")
st.markdown(
    "Retrains FORECASTING models (Ridge, Random Forest, Logistic, XGBoost, Neural Network) using "
    "only data available prior to the selected month. "
    "Saves to `models/forecasting/`."
)

if st.button("Retrain Forecasting Models", type="primary", key="btn_retrain_forecast"):
    from src.train_and_save import train_and_save_forecasting

    log_container = st.empty()
    logger = StreamlitLogger(log_container)

    with st.status("Training FORECASTING models...", expanded=True) as status:
        old_stdout = sys.stdout
        sys.stdout = logger
        try:
            train_and_save_forecasting()
            sys.stdout = old_stdout
            status.update(label="FORECASTING training complete.", state="complete", expanded=False)
            st.success("FORECASTING models retrained and saved successfully.")
        except Exception:
            sys.stdout = old_stdout
            status.update(label="FORECASTING training failed.", state="error", expanded=True)
            st.error("Training failed. See log above for details.")
            st.code(traceback.format_exc(), language="text")

    st.cache_data.clear()
    st.cache_resource.clear()
