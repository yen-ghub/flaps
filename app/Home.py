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
from src.ui_theme import apply_theme

st.set_page_config(
    page_title="FLAPS — Flight Delay Predictor",
    page_icon="✈️",
    layout="wide",
)

_FINDING_CARD_CSS = """
.finding-card {
    border: 1px solid rgba(13, 13, 13, 0.25);
    padding: 1.2rem 1.2rem 1rem 1.2rem;
    height: 100%;
}
.finding-card .kicker {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #b07d2e;
    margin-bottom: 0.4rem;
}
.finding-card h4 {
    font-size: 1rem;
    font-weight: 600;
    margin: 0 0 0.5rem 0;
    color: #0d0d0d;
    line-height: 1.35;
}
.finding-card p {
    font-size: 0.88rem;
    color: #5a5a5a;
    line-height: 1.5;
    margin: 0;
}
"""

apply_theme(extra_css=_FINDING_CARD_CSS)


@st.cache_data
def get_metadata():
    return load_metadata()


# Load metadata of the prediction model (model/), including the pre-trained parameters.
# If pre-trained parameters do not exist, prompt user to train the model using CLI
try:
    metadata = get_metadata()
except FileNotFoundError:
    st.error("Model artifacts not found. Run `python -m src.train_and_save` first.")
    st.stop()

# Display title
st.title("✈️ Project FLAPS")
st.markdown(
    """
    <div class="swiss-note">
        A monthly delay prediction model for Australian domestic flight routes.
    </div>
    """,
    unsafe_allow_html=True,
)
# 

# Insert a horizontal divider line
st.divider()

# --- About section ---
st.subheader('Why')
st.markdown("""
            The on-time performance of Australian domestic flights used to be at 81% (over the 2010-2019 period).

            Surprisingly, it had declined to 63-68% in 2023-2024. What is going on?

            On the premise that better predictions could lead to better planning for airlines and airports, the idea for Project FLAPS (Flight Lateness Australia Prediction System) was born. 
            """)

st.subheader('What')
st.markdown("""            
            The primary aim of Project FLAPS is to develop a machine learning model that can predict next month’s flight delays. 
            
            This has now taken form as the FORECASTING Model.
            """)

st.subheader('How')
st.markdown("""       
            As the secondary aim of Project FLAPS is a personal endeavour to build something useful from free and publicly accessible data, it uses:
            - flight performance data published by the Bureau of Infrastructure and Transport Research Economics ([BITRE](https://www.bitre.gov.au/))
            - weather observations data published by the Bureau of Meteorology ([BOM](http://www.bom.gov.au/)).
            """)
st.markdown("However, this led to a fundamental constraint -- " 
            "the data above are published monthly and usually only by the end of the first week of the _following_ month."
            )
st.markdown("This means that real-time forecasting is _not_ possible, as real-time data (i.e. current month's data) is not yet available to predict delays in the current month."
            )
# st.markdown("This creates a fundamental constraint:  \n"
#             "_current month's data is not available to predict current month's delays (i.e. real-time forecasting not possible)_."
#             )
st.markdown("""
            To address this constraint, the following two-part approach is used to evaluate 7 different machine learning models:  
            1. **NOWCASTING Models** 
                - Use data up until the previous month to explain the _previous_ month's flight delays.  
                - There is no forecast for the current month, however the purpose is to validate the model architecture and identify dominant features.
                - For example: it uses January 2026 data to _explain_ January 2026 delays.
            2. **FORECASTING Models**
                - Use data up until the previous month to predict the _current_ month's flight delays.
                - Forecast is available for the current month, usually by the end of the first week of the month.
                - For example: it uses January 2026 data to _predict_ February 2026 delays (by the first week of February 2026).
            """)
st.markdown("For both approaches, the best overall models are **Ridge** for regression modelling and **XGBoost** for classification modelling.")
st.markdown("All models are trained on 15 years of data (2010-2025, excluding the COVID period), and includes 21 routes and 6 airlines.")
st.markdown("Feature selection and engineering are performed following a hypothesis testing approach, where each potential feature is tested in isolation and retained only when a measurable improvement is observed.")

st.divider()

# --- Key metrics row ---
# col1, col2, col3, col4, col5, col6 = st.columns(6)

# with col1:
#     n_routes = len(metadata['valid_routes'])
#     st.metric("Routes", n_routes, help="Airline-route combinations in the production model (filtered to ≥50 flights/month)")

# with col2:
#     n_airlines = len(metadata['valid_airlines'])
#     st.metric("Airlines", n_airlines)

# with col3:
#     st.metric("Years of Data", "15", help="2010–2025, excluding the COVID period (2020–2022)")

# with col4:
#     best_r2 = metadata['metrics']['rf_reg']['R2']
#     st.metric("Best R²", f"{best_r2:.3f}", help="Random Forest regression on held-out test set (nowcasting)")

# with col5:
#     best_f1 = metadata['metrics']['xgb_clf']['F1']
#     st.metric("Best F1", f"{best_f1:.3f}", help="XGBoost classification on held-out test set (nowcasting)")

# with col6:
#     rf_r2 = metadata['metrics']['rf_reg']['R2']
#     baseline_r2 = metadata.get('baseline_lag1_r2')
#     if baseline_r2 is not None:
#         improvement = rf_r2 - baseline_r2
#         st.metric("Beats Baseline By", f"+{improvement:.3f}", help="R² improvement of Random Forest over the naive lag-1 baseline")
#     else:
#         st.metric("Beats Baseline By", "—")

# st.divider()


# --- Skills Demonstrated section ---
st.subheader("Skills Demonstrated")

def _skill(header, description):
    return (
        f"- **{header}**<br>"
        f'<span style="font-style:italic; color:#5a5a5a;">{description}</span>'
    )

_skills_left = [
    ("Data acquisition",                  "Automated pipelines for BOM FTP and BITRE web sources (one-click update)"),
    ("Feature engineering",               "Hand-engineered features from weather, holidays, past delays, and encoding"),
    ("Hypothesis-driven experimentation", "Systematic feature evaluation documented in notebooks for interpretability"),
    ("Time-series validation",            "COVID-aware training splits to prevent temporal leakage"),
]

_skills_right = [
    ("Model evaluation",      "5 models compared with cross-validation and baseline benchmarks"),
    ("Problem diagnosis",     "Identified volume-based filtering to exclude noisy data"),
    ("Model selection",       "Simpler models were preferred when added complexity did not justify the gains"),
    ("End-to-end deployment", "From raw data to web-deployed interactive application"),
]

col_a, col_b = st.columns(2)

with col_a:
    st.markdown(
        "\n".join(_skill(h, d) for h, d in _skills_left),
        unsafe_allow_html=True,
    )

with col_b:
    st.markdown(
        "\n".join(_skill(h, d) for h, d in _skills_right),
        unsafe_allow_html=True,
    )


