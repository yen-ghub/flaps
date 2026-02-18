"""
FLAPS — Flight Lateness Australia Prediction System
Streamlit home page.
"""

import sys
import os

# Add project root to path so the source codes (src/) can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.data_loader import load_metadata

st.set_page_config(
    page_title="FLAPS — Flight Delay Predictor",
    page_icon="✈️",
    layout="wide",
)

# Load metadata of the prediction model (model/), including the pre-trained parameters.
# If pre-trained parameters do not exist, prompt user to train the model using CLI
try:
    metadata = load_metadata()
except FileNotFoundError:
    st.error("Model artifacts not found. Run `python -m src.train_and_save` first.")
    st.stop()

# Display title
st.title("Flight Lateness Australia Prediction System")
st.markdown("Predicting monthly flight delay rates for Australian domestic routes using machine learning.")

# Insert a horizontal divider line
st.divider()

# Show key metrics
# Columns will divide layout into equal width
col1, col2, col3, col4 = st.columns(4)

with col1:
    n_routes = len(metadata['valid_routes'])
    st.metric("Routes", n_routes)

with col2:
    n_airlines = len(metadata['valid_airlines'])
    st.metric("Airlines", n_airlines)

with col3:
    best_r2 = metadata['metrics']['rf_reg']['R2']
    st.metric("Best R²", f"{best_r2:.3f}", help="Random Forest regression on test set")

with col4:
    best_f1 = metadata['metrics']['xgb_clf']['F1']
    st.metric("Best F1", f"{best_f1:.3f}", help="XGBoost classification on test set")

st.divider()

# Insert "About" section
st.subheader("About This Project")

st.markdown("""
Australian domestic flight punctuality declined to 63–68% in 2023–2024, well below the
81% long-term average. This project builds machine learning models to predict monthly
delay rates for route-airline combinations using publicly available data from
[BITRE](https://www.bitre.gov.au/) (flight performance) and
[BOM](http://www.bom.gov.au/) (weather observations).

The models are trained and tested on a combined 15 years of data (2010–2025, excluding COVID) across 21 routes
and 7 airlines. The primary finding is that the previous month's delay rate is the dominant
predictor — delays persist due to operational factors rather than weather conditions alone.
""")

# Insert "Skills" section
st.subheader("Skills Demonstrated")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
    - **Data acquisition** — automated pipelines for BOM FTP and BITRE web sources
    - **Feature engineering** — 38 features from weather, holidays, lag, and encoding
    - **Hypothesis-driven experimentation** — systematic feature evaluation (12 notebooks)
    """)

with col_b:
    st.markdown("""
    - **Model evaluation** — 5 models compared with cross-validation and baseline benchmarks
    - **Problem diagnosis** — identified volume-based filtering to improve R² by +0.09
    - **End-to-end deployment** — from raw data to interactive application
    """)

st.divider()

st.markdown("""
**Navigate** using the sidebar to explore predictions, model performance, key findings,
and interactive data exploration.
""")
