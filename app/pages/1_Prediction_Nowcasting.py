"""
Interactive prediction page.
Users select route, airline, and a specific year-month to get delay rate predictions
from all models and compare them against actual observed values.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Import necessary functions from source code
from src.data_loader import load_metadata, load_models, load_training_data
from src.feature_engineering import (
    add_derived_columns,
    compute_cyclical_month,
    compute_lag_features,
    compute_weather_transforms,
    filter_anomalous_routes,
    filter_low_volume,
)

st.set_page_config(page_title="FLAPS — Prediction: Nowcasting", page_icon="✈️", layout="wide")
st.title("Nowcasting")
st.markdown(
    """
    <style>
    :root {
        --flaps-surface: #efeee9;
        --flaps-card: #f8f7f3;
        --flaps-text: #0d0d0d;
        --flaps-muted: #5a5a5a;
        --flaps-border: rgba(13, 13, 13, 0.25);
        --flaps-accent: #0d0d0d;
    }
    .stApp {
        background: var(--flaps-surface);
        color: var(--flaps-text);
        font-family: "Helvetica Neue", "Nimbus Sans L", "Liberation Sans", sans-serif;
    }
    h1, h2, h3 {
        letter-spacing: -0.02em;
        color: var(--flaps-text);
    }
    [data-testid="stDivider"] {
        border-top: 1px solid var(--flaps-border);
    }
    .swiss-note {
        max-width: 900px;
        color: var(--flaps-muted);
        font-size: 1rem;
        line-height: 1.5;
        margin-bottom: 1.2rem;
    }
    [data-testid="stMetric"] {
        background: transparent;
        border: 1px solid var(--flaps-border);
        border-radius: 0;
        padding: 0.8rem;
        min-height: 132px;
    }
    [data-testid="stMetricLabel"] {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.72rem;
    }
    [data-testid="stMetricValue"] {
        white-space: normal;
        overflow: visible;
        text-overflow: clip;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.8rem;
    }
    [data-testid="stAlert"] {
        border-radius: 0;
        border: 1px solid var(--flaps-border);
    }
    [data-testid="stSidebar"] {
        background: var(--flaps-surface);
        border-right: 1px solid var(--flaps-border);
    }
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        background: var(--flaps-surface);
    }
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
        border-radius: 0 !important;
        color: var(--flaps-text);
        background: transparent !important;
        padding: 0.28rem 0.45rem;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
        background: transparent !important;
        text-decoration: underline;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {
        background: transparent !important;
        font-weight: 600;
        border-left: 2px solid var(--flaps-text);
        padding-left: calc(0.45rem - 2px);
    }
    [data-testid="stSidebar"] hr {
        border-color: var(--flaps-border);
        margin: 0.9rem 0;
    }
    [data-testid="stSidebar"] .stSelectbox label {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.72rem;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        border-radius: 0;
        border: 1px solid var(--flaps-border);
        background: var(--flaps-card);
        min-height: 42px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="swiss-note">
        Nowcasting uses data available up to and including the prediction month, which allows
        same-month context to be incorporated. Select a route, airline, and month to compare model
        predictions against observed outcomes.
    </div>
    """,
    unsafe_allow_html=True,
)

# Streamlit re-runs the entire script with every interaction (any click, dropdown, slider, etc.)
# To avoid re-loading the models and data with every user action, caching is used.
@st.cache_resource
def get_models():
    return load_models()

@st.cache_data
def get_metadata():
    return load_metadata()

@st.cache_data
def get_data():
    df = load_training_data()
    df = add_derived_columns(df)
    df = filter_low_volume(df)
    df = filter_anomalous_routes(df)
    df = compute_lag_features(df)
    df, _ = compute_weather_transforms(df)
    df = compute_cyclical_month(df)
    return df


# Actually load the models and data
models = get_models()
metadata = get_metadata()
df = get_data()

# Check which models are available
has_nn = 'nn_reg' in models and 'nn_clf' in models  # neural network
has_xgb = 'xgb_clf' in models                       # XGBoost

# --- Sidebar inputs ---
# Create a sidebar for user input (inference)
st.sidebar.header("Nowcasting Inputs")

# Route selection dropdown (show as "City -> City")
valid_routes = metadata['valid_routes']
route_labels = {r: r.replace('_', ' \u2192 ') for r in valid_routes}
selected_route = st.sidebar.selectbox(
    "Route",
    options=valid_routes,
    format_func=lambda r: route_labels[r],
    index=19,
)

# Airline selection dropdown (filtered to those operating on the selected route)
route_df = df[df['route'] == selected_route]
route_airlines = sorted(route_df['airline'].unique().tolist())
selected_airline = st.sidebar.selectbox("Airline", options=route_airlines)

# Combile the airline-route selections and filter data
airline_route = f"{selected_airline}_{selected_route}"
ar_data = df[df['airline_route'] == airline_route].sort_values('year_month_dt')

# Handle error when there is no available data for inference
if len(ar_data) == 0:
    st.warning("No historical data available for this airline-route combination.")
    st.stop()

# Year-month selection: manually limit to the last 12 months of the complete date range
# We need rows that have lag1 available (not NaN) so predictions can be made
ar_with_lag = ar_data.dropna(subset=['delay_rate_lag1'])
if len(ar_with_lag) == 0:
    st.warning("Insufficient data to compute lag features for this airline-route.")
    st.stop()

# Determine the last 12 calendar months from the full dataset (before any split)
all_months_sorted = sorted(df['year_month_dt'].unique())
last_12_cutoff = all_months_sorted[-12] if len(all_months_sorted) >= 12 else all_months_sorted[0]
available_months = sorted(ar_with_lag[
    ar_with_lag['year_month_dt'] >= last_12_cutoff
]['year_month_dt'].unique().tolist())

if len(available_months) == 0:
    st.warning("No data available for this airline-route in the last 12 months.")
    st.stop()

month_labels = {dt: pd.Timestamp(dt).strftime('%B %Y') for dt in available_months}

# Month selection dropdown
selected_dt = st.sidebar.selectbox(
    "Prediction Month",
    options=list(reversed(available_months)),
    format_func=lambda dt: month_labels[dt],
)

# --- Start collecting features to assemble input X ---
matched = ar_with_lag[ar_with_lag['year_month_dt'] == selected_dt]
if len(matched) == 0:
    # Handle edge case where Streamlit session state holds a stale value
    selected_dt = available_months[-1]
    matched = ar_with_lag[ar_with_lag['year_month_dt'] == selected_dt]
row = matched.iloc[0]

lag1_value = row['delay_rate_lag1']
actual_delay_rate = row['delay_rate']
actual_is_high = row['is_high_delay']
selected_month = int(row['month_num'])

# Get lag2 for gradient
if 'delay_rate_lag2' in row.index and pd.notna(row.get('delay_rate_lag2')):
    lag2_value = row['delay_rate_lag2']
else:
    # Fall back to computing from data
    idx = ar_data.index.get_loc(row.name)
    lag2_value = ar_data.iloc[idx - 1]['delay_rate'] if idx >= 1 else lag1_value

gradient = row.get('delay_rate_gradient', lag1_value - lag2_value)
if pd.isna(gradient):
    gradient = lag1_value - lag2_value

# Get lag12 for annual seasonality
target_lag12_dt = pd.Timestamp(selected_dt) - pd.DateOffset(months=12)
lag12_match = ar_data[ar_data['year_month_dt'] == target_lag12_dt]
if len(lag12_match) > 0:
    lag12_value = lag12_match.iloc[0]['delay_rate']
else:
    # Fallback: use lag1 (degrades performance but allows prediction)
    lag12_value = lag1_value

st.sidebar.divider()
st.sidebar.markdown(f"**Previous month delay rate (lag1):** {lag1_value:.1%}")

# --- Build feature vector ---
month_sin = np.sin(2 * np.pi * selected_month / 12)
month_cos = np.cos(2 * np.pi * selected_month / 12)

# Use the actual weather/holiday values from the row
feature_names = metadata['feature_names']
feature_values = {}

# One-hot encode the airlines
for col in metadata['airline_cols']:
    feature_values[col] = 1.0 if col == f"airline_{selected_airline}" else 0.0

# One-hot encode the routes
for col in metadata['route_cols']:
    feature_values[col] = 1.0 if col == f"route_{selected_route}" else 0.0

# Numeric features from the actual data row
feature_values['month_sin'] = month_sin
feature_values['month_cos'] = month_cos
feature_values['delay_rate_lag1'] = lag1_value
feature_values['sectors_scheduled'] = row['sectors_scheduled']
feature_values['rainy_days_arr_exp'] = row['rainy_days_arr_exp']
feature_values['delay_rate_lag12'] = lag12_value
feature_values['delay_rate_gradient'] = gradient
feature_values['temp_volatility_total_exp'] = row['temp_volatility_total_exp']
feature_values['extreme_weather_days_total'] = row['extreme_weather_days_total']
feature_values['n_public_holidays_total'] = row['n_public_holidays_total']
feature_values['pct_school_holiday'] = row['pct_school_holiday']

# Combine everything into input matrix X!
X = np.array([[feature_values[f] for f in feature_names]])



# --- Start prediction (nowcasting) ---
scaler = models['scaler']
X_scaled = scaler.transform(X)

# Regression predictions
ridge_pred = models['ridge'].predict(X_scaled)[0]
rf_pred = models['rf_reg'].predict(X)[0]
nn_reg_pred = models['nn_reg'].predict(X_scaled, verbose=0).flatten()[0] if has_nn else None

# Classification predictions
logreg_proba = models['logreg'].predict_proba(X_scaled)[0][1]
rf_clf_proba = models['rf_clf'].predict_proba(X)[0][1]
xgb_proba = models['xgb_clf'].predict_proba(X)[0][1] if has_xgb else None
nn_clf_proba = models['nn_clf'].predict(X_scaled, verbose=0).flatten()[0] if has_nn else None

selected_label = month_labels[selected_dt]

# Assemble model outputs for both brief summary and detailed sections
reg_models = [("Ridge", ridge_pred), ("Random Forest", rf_pred)]
if has_nn:
    reg_models.append(("Neural Network", nn_reg_pred))
ensemble_reg_pred = float(np.mean([pred for _, pred in reg_models]))

clf_models = [("Logistic", logreg_proba), ("Random Forest", rf_clf_proba)]
if has_xgb:
    clf_models.append(("XGBoost", xgb_proba))
if has_nn:
    clf_models.append(("Neural Network", nn_clf_proba))

avg_proba = float(np.mean([proba for _, proba in clf_models]))

# --- Display results ---
st.subheader("Nowcasting Brief")
brief_cols = st.columns(4)

with brief_cols[0]:
    st.metric("Route", route_labels[selected_route])
with brief_cols[1]:
    st.metric("Airline", selected_airline)
with brief_cols[2]:
    st.metric("Prediction Month", selected_label)
with brief_cols[3]:
    st.metric("Ensemble Delay Rate", f"{ensemble_reg_pred:.1%}")

st.divider()

st.subheader("Delay Rate: Predicted vs Actual")

n_reg_cols = len(reg_models) + 1  # +1 for actual
cols = st.columns(n_reg_cols)

# Displav regression predictions in columns
for i, (name, pred) in enumerate(reg_models):
    with cols[i]:
        err = pred - actual_delay_rate
        st.metric(
            name,
            f"{pred:.1%}",
            delta=f"{err:+.1%} error",
            delta_color="inverse",
        )

# The actual delay rate from BITRE
with cols[-1]:
    st.metric(
        "Actual Delay Rate",
        f"{actual_delay_rate:.1%}",
        help=f"Observed delay rate for {selected_label}",
    )

st.divider()

st.subheader("High-Delay Probability: Predicted vs Actual")
st.caption("Probability that the delay rate exceeds 25%")

actual_label = "Yes" if actual_is_high else "No"

n_clf_cols = len(clf_models) + 1  # +1 for actual
cols = st.columns(n_clf_cols)

# Display classification predictions
for i, (name, proba) in enumerate(clf_models):
    with cols[i]:
        correct = (proba >= 0.5) == bool(actual_is_high)
        st.metric(
            name,
            f"{proba:.1%}",
            delta="Matches actual" if correct else "Mismatch",
            delta_color="normal" if correct else "inverse",
        )

with cols[-1]:
    st.metric(
        "Actual High-Delay",
        actual_label,
        help=f"delay_rate = {actual_delay_rate:.1%}, threshold = 25%",
    )

# Gauge chart for average classifier prediction
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=avg_proba * 100,
    number={'suffix': '%'},
    gauge={
        'axis': {'range': [0, 100]},
        'bgcolor': "#efeee9",
        'bar': {'color': "#145da0"},
        'steps': [
            {'range': [0, 30], 'color': "#d8f3e6"},
            {'range': [30, 60], 'color': "#fff4d6"},
            {'range': [60, 100], 'color': "#fee4e2"},
        ],
        'threshold': {
            'line': {'color': "#b42318", 'width': 3},
            'thickness': 0.75,
            'value': 50,
        },
    },
    title={'text': "High-Delay Probability (Ensemble Average)"},
))
fig.update_layout(
    height=300,
    margin=dict(t=60, b=20, l=40, r=40),
    paper_bgcolor="#efeee9",
    plot_bgcolor="#efeee9",
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# Context
st.subheader("Prediction Context")
context_col1, context_col2 = st.columns(2)
with context_col1:
    st.write(f"**Month:** {selected_label}")
    st.write(f"**Previous month delay rate (lag1):** {lag1_value:.1%}")
    st.write(f"**Same month last year (lag12):** {lag12_value:.1%}")
    st.write(
        f"**Delay rate gradient:** {gradient:+.1%} "
        f"({'improving' if gradient < 0 else 'worsening' if gradient > 0 else 'stable'})"
    )
with context_col2:
    st.write(f"**Scheduled sectors:** {int(row['sectors_scheduled'])}")
    st.write(f"**Rainy days at arrival (exp):** {row['rainy_days_arr_exp']:.2f}")
    st.write(f"**Extreme weather days (total):** {row['extreme_weather_days_total']:.1f}")
