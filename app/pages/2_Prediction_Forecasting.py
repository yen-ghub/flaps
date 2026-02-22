"""
Forecasting prediction page.
Predicts delay rate using only historically available data.
This means no same-month data is used as features, including the weather data.

The forecasting page contains two sections:
1. Next Month Prediction: genuine future-looking (next month) forecast (no actual values available)
2. Past 12 Month Performance: to assess the forecasting model accuracy in the past 12 months
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import (
    load_forecasting_metadata,
    load_forecasting_models,
    load_load_factor_data,
    load_training_data,
)
from src.ui_theme import apply_theme
from src.feature_engineering import (
    add_derived_columns,
    compute_cyclical_month,
    compute_lag_features,
    compute_load_factor_features,
    compute_weather_transforms,
    filter_anomalous_routes,
    filter_low_volume,
)

st.set_page_config(page_title="FLAPS — Prediction: Forecasting", page_icon="✈️", layout="wide")
apply_theme()
st.title("Forecasting")
st.markdown(
    """
    <div class="swiss-note">
        <p>Predicts delay rate using only historically available data.
        This means no same-month data is used as features, including weather data.</p>
        <p>The forecasting page contains two sections:</p>
        <ol>
            <li>Next Month Prediction: genuine future-looking (next month) forecast (no actual values available)</li>
            <li>Past 12 Month Performance: assess forecasting model accuracy over the past 12 months</li>
        </ol>
    </div>
    """,
    unsafe_allow_html=True,
)



@st.cache_resource
def get_models():
    return load_forecasting_models()


@st.cache_data
def get_metadata():
    return load_forecasting_metadata()


@st.cache_data
def get_load_factor():
    """Load the market-wide monthly load factor data. Returns None if unavailable."""
    try:
        return load_load_factor_data()
    except FileNotFoundError:
        return None


@st.cache_data
def get_data():
    df = load_training_data()
    df = add_derived_columns(df)
    # Ensure year_month_dt is proper datetime (CSV may store as string)
    df['year_month_dt'] = pd.to_datetime(df['year_month_dt'])
    df = filter_low_volume(df)
    df = filter_anomalous_routes(df)
    df = compute_lag_features(df)
    df, _ = compute_weather_transforms(df)
    df = compute_cyclical_month(df)
    # Forecasting: add lag12
    df = df.copy()
    df['delay_rate_lag12'] = df.groupby('airline_route')['delay_rate'].shift(12)
    # Load factor features (if data available)
    df_lf = get_load_factor()
    if df_lf is not None:
        df = df.merge(df_lf[['year_month', 'load_factor']], on='year_month', how='left')
        df = compute_load_factor_features(df)
    return df


# Load
try:
    models = get_models()
    metadata = get_metadata()
except FileNotFoundError:
    st.error(
        "Forecasting models not found. Please train them first via the "
        "Update & Train page or by running: `python -m src.train_and_save --forecasting`"
    )
    st.stop()

df = get_data()

# Check which models are available
has_xgb = 'xgb_clf' in models
has_nn = 'nn_reg' in models and 'nn_clf' in models

# --- Sidebar inputs (route and airline only) ---
st.sidebar.header("Forecasting Inputs")

# Route selection (show as "City -> City")
valid_routes = metadata['valid_routes']
route_labels = {r: r.replace('_', ' \u2192 ') for r in valid_routes}
selected_route = st.sidebar.selectbox(
    "Route",
    options=valid_routes,
    format_func=lambda r: route_labels[r],
    index=min(19, len(valid_routes) - 1),
)

# Airline selection (filtered to those operating on the selected route)
route_df = df[df['route'] == selected_route]
route_airlines = sorted(route_df['airline'].unique().tolist())
selected_airline = st.sidebar.selectbox("Airline", options=route_airlines)

# Filter to this airline-route
airline_route = f"{selected_airline}_{selected_route}"
ar_data = df[df['airline_route'] == airline_route].sort_values('year_month_dt')

if len(ar_data) == 0:
    st.warning("No historical data available for this airline-route combination.")
    st.stop()


# ── Helper: build feature vector and predict ──

def build_features_and_predict(month_num, lag1, lag12, gradient, sectors, holidays_total,
                               pct_school, load_factor_lag1_exp=None):
    """Build the feature vector and return predictions from all models."""
    month_sin = np.sin(2 * np.pi * month_num / 12)
    month_cos = np.cos(2 * np.pi * month_num / 12)

    feature_names = metadata['feature_names']
    feature_values = {}

    # One-hot airline
    for col in metadata['airline_cols']:
        feature_values[col] = 1.0 if col == f"airline_{selected_airline}" else 0.0

    # One-hot route
    for col in metadata['route_cols']:
        feature_values[col] = 1.0 if col == f"route_{selected_route}" else 0.0

    # Numeric features
    feature_values['month_sin'] = month_sin
    feature_values['month_cos'] = month_cos
    feature_values['delay_rate_lag1'] = lag1
    feature_values['sectors_scheduled'] = sectors
    feature_values['delay_rate_lag12'] = lag12
    feature_values['delay_rate_gradient'] = gradient
    feature_values['n_public_holidays_total'] = holidays_total
    feature_values['pct_school_holiday'] = pct_school

    # Load factor (LF_exp) — only included if model was trained with it
    if 'load_factor_lag1_exp' in feature_names:
        # Use provided value; fall back to 1.0 (exp(0)) if unavailable
        feature_values['load_factor_lag1_exp'] = (
            float(load_factor_lag1_exp) if load_factor_lag1_exp is not None else 1.0
        )

    X = np.array([[feature_values[f] for f in feature_names]])

    scaler = models['scaler']
    X_scaled = scaler.transform(X)

    ridge_pred = models['ridge'].predict(X_scaled)[0]
    rf_pred = models['rf_reg'].predict(X)[0]
    logreg_proba = models['logreg'].predict_proba(X_scaled)[0][1]
    rf_clf_proba = models['rf_clf'].predict_proba(X)[0][1]
    xgb_proba = models['xgb_clf'].predict_proba(X)[0][1] if has_xgb else None
    nn_reg_pred = float(models['nn_reg'].predict(X_scaled, verbose=0).flatten()[0]) if has_nn else None
    nn_clf_proba = float(models['nn_clf'].predict(X_scaled, verbose=0).flatten()[0]) if has_nn else None

    return {
        'ridge': ridge_pred, 'rf': rf_pred,
        'logreg': logreg_proba, 'rf_clf': rf_clf_proba, 'xgb': xgb_proba,
        'nn_reg': nn_reg_pred, 'nn_clf': nn_clf_proba,
    }


def render_gauge(avg_proba):
    """Render the high-delay probability gauge chart."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Next Month Prediction
# ═══════════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("Next Month Prediction")

# Compute the next month after latest data
latest_dt = pd.Timestamp(df['year_month_dt'].max())
next_month_dt = latest_dt + pd.DateOffset(months=1)
next_month_label = next_month_dt.strftime('%B %Y')
next_month_num = next_month_dt.month

st.markdown(f'<div class="swiss-section-kicker">Predicting: {next_month_label}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="swiss-note">'
    "The next month is forecast from currently available history only. "
    "Observed outcomes are not yet available."
    '</div>',
    unsafe_allow_html=True,
)

# Build next-month features from historical data
ar_sorted = ar_data.sort_values('year_month_dt')
latest_row = ar_sorted.iloc[-1]

# lag1 = latest month's delay rate
next_lag1 = latest_row['delay_rate']

# lag2 = second-to-last month's delay rate (for gradient)
if len(ar_sorted) >= 2:
    next_lag2 = ar_sorted.iloc[-2]['delay_rate']
else:
    next_lag2 = next_lag1
next_gradient = next_lag1 - next_lag2

# lag12 = same calendar month last year
target_lag12_dt = next_month_dt - pd.DateOffset(months=12)
lag12_match = ar_sorted[ar_sorted['year_month_dt'] == target_lag12_dt]

# Proxy features from same month last year
same_month_ly = ar_sorted[ar_sorted['year_month_dt'] == target_lag12_dt]

can_predict_next = True

if len(lag12_match) == 0:
    st.warning(
        f"Cannot predict {next_month_label}: no data for {target_lag12_dt.strftime('%B %Y')} "
        f"(needed for lag12 feature). This airline-route may not have been operating 12 months ago."
    )
    can_predict_next = False

if can_predict_next:
    next_lag12 = lag12_match.iloc[0]['delay_rate']
    next_sectors = latest_row['sectors_scheduled']
    next_holidays = same_month_ly.iloc[0]['n_public_holidays_total']
    next_pct_school = same_month_ly.iloc[0]['pct_school_holiday']

    # load_factor_lag1_exp for next month = exp(most recent available load_factor).
    # We use the latest value from df_lf directly rather than latest_row['load_factor']
    # because the training data may extend past the LF file's coverage (e.g. Dec 2025
    # exists in training data but Nov 2025 is the latest in the LF Excel file).
    next_lf_exp = None
    df_lf = get_load_factor()
    if df_lf is not None and len(df_lf) > 0:
        latest_lf = df_lf.sort_values('year_month').iloc[-1]['load_factor']
        if pd.notna(latest_lf):
            next_lf_exp = float(np.exp(latest_lf))

    preds = build_features_and_predict(
        month_num=next_month_num,
        lag1=next_lag1,
        lag12=next_lag12,
        gradient=next_gradient,
        sectors=next_sectors,
        holidays_total=next_holidays,
        pct_school=next_pct_school,
        load_factor_lag1_exp=next_lf_exp,
    )
    next_all_proba = [preds['logreg'], preds['rf_clf']]
    if has_xgb:
        next_all_proba.append(preds['xgb'])
    if has_nn and preds['nn_clf'] is not None:
        next_all_proba.append(preds['nn_clf'])
    next_avg_proba = float(np.mean(next_all_proba))
    reg_preds = [preds['ridge'], preds['rf']]
    if has_nn and preds['nn_reg'] is not None:
        reg_preds.append(preds['nn_reg'])
    next_ensemble_reg_pred = float(np.mean(reg_preds))

    st.markdown("**Forecasting Brief**")
    brief_cols = st.columns(4)
    with brief_cols[0]:
        st.metric("Route", route_labels[selected_route])
    with brief_cols[1]:
        st.metric("Airline", selected_airline)
    with brief_cols[2]:
        st.metric("Prediction Month", next_month_label)
    with brief_cols[3]:
        st.metric("Ensemble Delay Rate", f"{next_ensemble_reg_pred:.1%}")

    st.divider()

    # Regression results (no actual, no error)
    st.markdown("**Delay Rate: Predicted**")
    reg_items = [("Ridge", preds['ridge']), ("Random Forest", preds['rf'])]
    if has_nn and preds['nn_reg'] is not None:
        reg_items.append(("Neural Network", preds['nn_reg']))
    reg_cols = st.columns(len(reg_items))
    for i, (name, pred) in enumerate(reg_items):
        with reg_cols[i]:
            st.metric(name, f"{pred:.1%}")

    st.divider()

    # Classification results (no actual, no correct/incorrect)
    st.markdown("**High-Delay Probability: Predicted**")
    st.caption("Probability that the delay rate exceeds 25%")

    clf_items = [("Logistic", preds['logreg']), ("Random Forest", preds['rf_clf'])]
    if has_xgb:
        clf_items.append(("XGBoost", preds['xgb']))
    if has_nn and preds['nn_clf'] is not None:
        clf_items.append(("Neural Network", preds['nn_clf']))

    clf_cols = st.columns(len(clf_items))
    for i, (name, proba) in enumerate(clf_items):
        with clf_cols[i]:
            st.metric(name, f"{proba:.1%}")

    # Gauge
    render_gauge(next_avg_proba)

    # Context
    st.markdown("**Prediction Context**")
    next_context_col1, next_context_col2 = st.columns(2)
    with next_context_col1:
        st.write(f"**Month:** {next_month_label}")
        st.write(f"**Previous month delay rate (lag1):** {next_lag1:.1%}")
        st.write(f"**Same month last year (lag12):** {next_lag12:.1%}")
        st.write(
            f"**Delay rate gradient:** {next_gradient:+.1%} "
            f"({'improving' if next_gradient < 0 else 'worsening' if next_gradient > 0 else 'stable'})"
        )
    with next_context_col2:
        st.write(f"**Scheduled sectors:** {int(next_sectors)} (previous month's value)")
        st.write(f"**Public holidays (total):** {next_holidays:.0f} (same month last year)")
        st.write(f"**School holiday coverage:** {next_pct_school:.0%} (same month last year)")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Past 12 Month Performance
# ═══════════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("Past 12 Month Performance")
st.caption(
    "Select a past month to compare model predictions against observed delay rates. "
    "This section assesses forecasting accuracy on historical data."
)

# Rows that have both lag1 and lag12 available
ar_with_lags = ar_data.dropna(subset=['delay_rate_lag1', 'delay_rate_lag12'])
if len(ar_with_lags) == 0:
    st.warning("Insufficient data to compute lag features (lag1 + lag12) for this airline-route.")
    st.stop()

# Last 12 calendar months from the full dataset
all_months_sorted = sorted(df['year_month_dt'].unique())
last_12_cutoff = all_months_sorted[-12] if len(all_months_sorted) >= 12 else all_months_sorted[0]
available_months = sorted(ar_with_lags[
    ar_with_lags['year_month_dt'] >= last_12_cutoff
]['year_month_dt'].unique().tolist())

if len(available_months) == 0:
    st.warning("No data available for this airline-route in the last 12 months.")
    st.stop()

month_labels = {dt: pd.Timestamp(dt).strftime('%B %Y') for dt in available_months}

selected_dt = st.selectbox(
    "Performance Month",
    options=list(reversed(available_months)),
    format_func=lambda dt: month_labels[dt],
    width=300,
)

# Look up the actual row for this year-month
matched = ar_with_lags[ar_with_lags['year_month_dt'] == selected_dt]
if len(matched) == 0:
    selected_dt = available_months[-1]
    matched = ar_with_lags[ar_with_lags['year_month_dt'] == selected_dt]
row = matched.iloc[0]

lag1_value = row['delay_rate_lag1']
lag12_value = row['delay_rate_lag12']
actual_delay_rate = row['delay_rate']
actual_is_high = row['is_high_delay']
selected_month = int(row['month_num'])

# Get gradient
if 'delay_rate_gradient' in row.index and pd.notna(row.get('delay_rate_gradient')):
    gradient = row['delay_rate_gradient']
else:
    lag2_value = row.get('delay_rate_lag2', lag1_value)
    if pd.isna(lag2_value):
        lag2_value = lag1_value
    gradient = lag1_value - lag2_value

selected_label = month_labels[selected_dt]

# Get load_factor_lag1_exp from the historical row (pre-computed by get_data())
past_lf_exp = None
if 'load_factor_lag1_exp' in row.index and pd.notna(row.get('load_factor_lag1_exp')):
    past_lf_exp = float(row['load_factor_lag1_exp'])

# Predict
past_preds = build_features_and_predict(
    month_num=selected_month,
    lag1=lag1_value,
    lag12=lag12_value,
    gradient=gradient,
    sectors=row['sectors_scheduled'],
    holidays_total=row['n_public_holidays_total'],
    pct_school=row['pct_school_holiday'],
    load_factor_lag1_exp=past_lf_exp,
)
reg_preds_past = [past_preds['ridge'], past_preds['rf']]
if has_nn and past_preds['nn_reg'] is not None:
    reg_preds_past.append(past_preds['nn_reg'])
past_ensemble_reg_pred = float(np.mean(reg_preds_past))
st.subheader("Performance Brief")
perf_cols = st.columns(4)
with perf_cols[0]:
    st.metric("Route", route_labels[selected_route])
with perf_cols[1]:
    st.metric("Airline", selected_airline)
with perf_cols[2]:
    st.metric("Performance Month", selected_label)
with perf_cols[3]:
    st.metric("Ensemble Delay Rate", f"{past_ensemble_reg_pred:.1%}")

st.divider()

# Regression results (with actual and error)
st.markdown("**Delay Rate: Predicted vs Actual**")

reg_models = [("Ridge", past_preds['ridge']), ("Random Forest", past_preds['rf'])]
if has_nn and past_preds['nn_reg'] is not None:
    reg_models.append(("Neural Network", past_preds['nn_reg']))
cols = st.columns(len(reg_models) + 1)

for i, (name, pred) in enumerate(reg_models):
    with cols[i]:
        err = pred - actual_delay_rate
        st.metric(
            name,
            f"{pred:.1%}",
            delta=f"{err:+.1%} error",
            delta_color="inverse",
        )

with cols[-1]:
    st.metric(
        "Actual Delay Rate",
        f"{actual_delay_rate:.1%}",
        help=f"Observed delay rate for {selected_label}",
    )

st.divider()

# Classification results (with actual and correct/incorrect)
st.markdown("**High-Delay Probability: Predicted vs Actual**")
st.caption("Probability that the delay rate exceeds 25%")

actual_label = "Yes" if actual_is_high else "No"

clf_models = [("Logistic", past_preds['logreg']), ("Random Forest", past_preds['rf_clf'])]
if has_xgb:
    clf_models.append(("XGBoost", past_preds['xgb']))
if has_nn and past_preds['nn_clf'] is not None:
    clf_models.append(("Neural Network", past_preds['nn_clf']))

cols = st.columns(len(clf_models) + 1)

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

# Gauge
past_all_proba = [past_preds['logreg'], past_preds['rf_clf']]
if has_xgb:
    past_all_proba.append(past_preds['xgb'])
if has_nn and past_preds['nn_clf'] is not None:
    past_all_proba.append(past_preds['nn_clf'])
render_gauge(np.mean(past_all_proba))

# Context
st.markdown("**Prediction Context**")
past_context_col1, past_context_col2 = st.columns(2)
with past_context_col1:
    st.write(f"**Month:** {selected_label}")
    st.write(f"**Previous month delay rate (lag1):** {lag1_value:.1%}")
    st.write(f"**Same month last year (lag12):** {lag12_value:.1%}")
    st.write(
        f"**Delay rate gradient:** {gradient:+.1%} "
        f"({'improving' if gradient < 0 else 'worsening' if gradient > 0 else 'stable'})"
    )
with past_context_col2:
    st.write(f"**Scheduled sectors:** {int(row['sectors_scheduled'])}")
    st.write(f"**Public holidays (total):** {row['n_public_holidays_total']:.0f}")
    st.write(f"**School holiday coverage:** {row['pct_school_holiday']:.0%}")
