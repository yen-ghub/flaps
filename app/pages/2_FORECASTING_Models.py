"""
Forecasting prediction page.
Predicts delay rate using only historically available data.
This means no same-month data is used as features, including the weather data.
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
    FORECASTING_MODELS_DIR,
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
st.markdown(
    """
    <style>
    [data-testid="stMetric"] {
        border-left: 3px solid #b07d2e;
        padding-left: calc(0.8rem - 2px);
    }
    div[class*="e12zf7d53"] {
        border-width: 3px !important;
        border-radius: 0 !important;
        border-color: rgba(13, 13, 13, 0.25) !important;
        padding-bottom: 0.6rem !important;
    }
    .stSelectbox label,
    .stSelectbox label * {
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        font-size: 0.9rem !important;
        font-weight: 400 !important;
    }
    .stSelectbox [data-baseweb="select"] div[class*="ValueContainer"] div,
    .stSelectbox [data-baseweb="select"] div[class*="singleValue"],
    .stSelectbox [data-baseweb="select"] div[class*="placeholder"] {
        font-size: 0.72rem !important;
    }
    .info-box {
        border: 1px solid rgba(13, 13, 13, 0.25);
        border-radius: 0;
        padding: 0.8rem;
        min-height: 132px;
        box-sizing: border-box;
    }
    .info-box .label {
        font-family: inherit;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.72rem;
        color: rgb(49, 51, 63);
        font-weight: 400;
        margin-bottom: 0.5rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .info-box .value {
        font-family: inherit;
        font-size: clamp(1.5rem, 2.5vw, 2.5rem);
        font-weight: 400;
        color: rgb(49, 51, 63);
        line-height: 1;
    }
    .actual-box {
        background-color: #e8e6df;
        border: 1px solid rgba(13, 13, 13, 0.25);
        border-radius: 0;
        padding: 0.8rem;
        min-height: 132px;
        height: 132px;
        box-sizing: border-box;
        overflow: hidden;
    }
    .actual-box .label {
        font-family: inherit;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.875rem;
        color: rgb(49, 51, 63);
        font-weight: 400;
        margin-bottom: 0.5rem;
    }
    .actual-box .value {
        font-family: inherit;
        font-size: clamp(1.5rem, 2.5vw, 2.5rem);
        font-weight: 400;
        color: rgb(49, 51, 63);
        line-height: 1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Forecasting Models")
st.markdown(
    """
    The forecasting models predict the delay rate of the current month, using data up until the month prior (i.e. no real-time data from the selected month is used).

    For example: when data is available up to February 2026, Forecasting prediction is available up to March 2026.

    _*The \"Actual Delay Rate\" is shown for comparison purposes only and is not used in making any predictions._
    """
)


@st.cache_resource
def get_fast_models():
    import joblib
    files = {
        'scaler': 'scaler.pkl',
        'ridge': 'ridge_regressor.pkl',
        'logreg': 'logreg_classifier.pkl',
        'xgb_clf': 'xgb_classifier.pkl',
    }
    return {k: joblib.load(os.path.join(FORECASTING_MODELS_DIR, f)) for k, f in files.items()}

@st.cache_resource
def get_slow_models():
    m = load_forecasting_models()
    return {k: m[k] for k in ('rf_reg', 'rf_clf', 'nn_reg', 'nn_clf') if k in m}


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
    df['year_month_dt'] = pd.to_datetime(df['year_month_dt'], format='mixed')
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


# Load metadata and training data eagerly (fast)
try:
    metadata = get_metadata()
except FileNotFoundError:
    st.error(
        "Forecasting models not found. Please train them first via the "
        "Update & Train page or by running: `python -m src.train_and_save --forecasting`"
    )
    st.stop()

df = get_data()

# Route/airline data setup
valid_routes = metadata['valid_routes']
route_labels = {r: r.replace('_', ' \u2192 ') for r in valid_routes}


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


# ── Compute next month ──
latest_dt = pd.Timestamp(df['year_month_dt'].max())
next_month_dt = latest_dt + pd.DateOffset(months=1)
next_month_label = next_month_dt.strftime('%B %Y')


# ── Select a Flight (unified) ──
st.divider()
st.subheader("Select a Flight")

with st.container(border=True):
    brief_cols = st.columns(3)

    with brief_cols[0]:
        selected_route = st.selectbox(
            "Route",
            options=valid_routes,
            format_func=lambda r: route_labels[r],
            index=min(19, len(valid_routes) - 1),
        )

    route_df = df[df['route'] == selected_route]
    route_airlines = sorted(
        a for a in route_df['airline'].unique()
        if a not in ('Virgin Australia Regional Airlines', 'Rex Airlines', 'Regional Express', 'Tigerair Australia')
    )

    with brief_cols[1]:
        selected_airline = st.selectbox("Airline", options=route_airlines)

    # Compute ar_data and available past months inside the container so we can
    # build the combined month list before rendering the Month selectbox.
    airline_route = f"{selected_airline}_{selected_route}"
    ar_data = df[df['airline_route'] == airline_route].sort_values('year_month_dt')

    ar_with_lags = ar_data.dropna(subset=['delay_rate_lag1', 'delay_rate_lag12'])
    all_months_sorted = sorted(df['year_month_dt'].unique())
    last_12_cutoff = all_months_sorted[-12] if len(all_months_sorted) >= 12 else all_months_sorted[0]
    available_months = sorted(ar_with_lags[
        ar_with_lags['year_month_dt'] >= last_12_cutoff
    ]['year_month_dt'].unique().tolist())

    # Combined list: next month first, then past 12 newest-first
    combined_months = [next_month_dt] + list(reversed(available_months))
    month_display = {dt: pd.Timestamp(dt).strftime('%B %Y') for dt in combined_months}

    with brief_cols[2]:
        selected_dt = st.selectbox(
            "Month",
            options=combined_months,
            format_func=lambda dt: month_display[dt],
        )

# Guard clauses
if len(ar_data) == 0:
    st.warning("No historical data available for this airline-route combination.")
    st.stop()
if len(ar_with_lags) == 0:
    st.warning("Insufficient data to compute lag features (lag1 + lag12) for this airline-route.")
    st.stop()

st.markdown(f"Latest available data: **{latest_dt.strftime('%B %Y')}**")

# ── Load models ──
with st.spinner("Loading models..."):
    models = {**get_fast_models(), **get_slow_models()}

has_xgb = 'xgb_clf' in models
has_nn = 'nn_reg' in models and 'nn_clf' in models

# ── Feature extraction (branches on future vs past) ──
is_future = (pd.Timestamp(selected_dt) == next_month_dt)

if is_future:
    ar_sorted = ar_data.sort_values('year_month_dt')
    latest_row = ar_sorted.iloc[-1]

    next_lag1 = latest_row['delay_rate']
    next_lag2 = ar_sorted.iloc[-2]['delay_rate'] if len(ar_sorted) >= 2 else next_lag1
    next_gradient = next_lag1 - next_lag2

    target_lag12_dt = next_month_dt - pd.DateOffset(months=12)
    lag12_match = ar_sorted[ar_sorted['year_month_dt'] == target_lag12_dt]
    same_month_ly = lag12_match

    if len(lag12_match) == 0:
        st.warning(
            f"Cannot predict {next_month_label}: no data for {target_lag12_dt.strftime('%B %Y')} "
            f"(needed for lag12 feature). This airline-route may not have been operating 12 months ago."
        )
        st.stop()

    next_lag12 = lag12_match.iloc[0]['delay_rate']
    next_sectors = latest_row['sectors_scheduled']
    next_holidays = same_month_ly.iloc[0]['n_public_holidays_total']
    next_pct_school = same_month_ly.iloc[0]['pct_school_holiday']

    next_lf_exp = None
    df_lf = get_load_factor()
    if df_lf is not None and len(df_lf) > 0:
        latest_lf = df_lf.sort_values('year_month').iloc[-1]['load_factor']
        if pd.notna(latest_lf):
            next_lf_exp = float(np.exp(latest_lf))

    preds = build_features_and_predict(
        month_num=next_month_dt.month,
        lag1=next_lag1,
        lag12=next_lag12,
        gradient=next_gradient,
        sectors=next_sectors,
        holidays_total=next_holidays,
        pct_school=next_pct_school,
        load_factor_lag1_exp=next_lf_exp,
    )
    actual_delay_rate = None
    actual_is_high = None
    selected_label = next_month_label
    lag1_value, lag12_value, gradient = next_lag1, next_lag12, next_gradient
    display_sectors = int(next_sectors)
    display_holidays = next_holidays
    display_pct_school = next_pct_school

else:
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

    if 'delay_rate_gradient' in row.index and pd.notna(row.get('delay_rate_gradient')):
        gradient = row['delay_rate_gradient']
    else:
        lag2_value = row.get('delay_rate_lag2', lag1_value)
        if pd.isna(lag2_value):
            lag2_value = lag1_value
        gradient = lag1_value - lag2_value

    selected_label = month_display[selected_dt]
    past_lf_exp = None
    if 'load_factor_lag1_exp' in row.index and pd.notna(row.get('load_factor_lag1_exp')):
        past_lf_exp = float(row['load_factor_lag1_exp'])

    preds = build_features_and_predict(
        month_num=selected_month,
        lag1=lag1_value,
        lag12=lag12_value,
        gradient=gradient,
        sectors=row['sectors_scheduled'],
        holidays_total=row['n_public_holidays_total'],
        pct_school=row['pct_school_holiday'],
        load_factor_lag1_exp=past_lf_exp,
    )
    display_sectors = int(row['sectors_scheduled'])
    display_holidays = row['n_public_holidays_total']
    display_pct_school = row['pct_school_holiday']


# ── Unified prediction output ──
st.divider()
st.subheader("Predictions")

# Regression results
st.markdown("**Delay rate**")
st.caption("Percentage of delayed flights in the selected month")

reg_items = [("Ridge", preds['ridge']), ("Random Forest", preds['rf'])]
if has_nn and preds['nn_reg'] is not None:
    reg_items.append(("Neural Network ★", preds['nn_reg']))

cols = st.columns(len(reg_items) + 1)
for i, (name, pred) in enumerate(reg_items):
    with cols[i]:
        if is_future:
            st.metric(f"Model: {name}", f"{pred:.1%}")
        else:
            err = pred - actual_delay_rate
            st.metric(f"Model: {name}", f"{pred:.1%}", delta=f"{err:+.1%} error", delta_color="inverse")

with cols[-1]:
    if is_future:
        actual_dr_display = "N/A"
    else:
        actual_dr_display = f"{actual_delay_rate:.1%}"
    st.markdown(
        f'<div class="actual-box">'
        f'<div class="label">Actual Delay Rate*</div>'
        f'<div class="value">{actual_dr_display}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.divider()

# Classification results
st.markdown("**High-Delay Month Probability**")
st.caption("Probability that more than 25% of the flights in the selected month are delayed")

clf_items = [("Logistic", preds['logreg']), ("Random Forest", preds['rf_clf'])]
if has_xgb:
    clf_items.insert(0, (" XGBoost ★", preds['xgb']))
if has_nn and preds['nn_clf'] is not None:
    clf_items.append(("Neural Network", preds['nn_clf']))

cols = st.columns(len(clf_items) + 1)
for i, (name, proba) in enumerate(clf_items):
    with cols[i]:
        if is_future:
            st.metric(f"Model: {name}", f"{proba:.1%}")
        else:
            correct = (proba >= 0.5) == bool(actual_is_high)
            st.metric(
                f"Model: {name}",
                f"{proba:.1%}",
                delta="Matches actual" if correct else "Mismatch",
                delta_color="normal" if correct else "inverse",
            )

with cols[-1]:
    if is_future:
        actual_hd_display = "N/A"
    else:
        actual_hd_display = "Yes" if actual_is_high else "No"
    st.markdown(
        f'<div class="actual-box">'
        f'<div class="label">Actual High-Delay</div>'
        f'<div class="value">{actual_hd_display}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# Gauge
all_proba = [preds['logreg'], preds['rf_clf']]
if has_xgb:
    all_proba.append(preds['xgb'])
if has_nn and preds['nn_clf'] is not None:
    all_proba.append(preds['nn_clf'])
render_gauge(float(np.mean(all_proba)))

# Prediction Context
st.divider()
st.subheader("Prediction Context")
ctx_col1, ctx_col2 = st.columns(2)
with ctx_col1:
    st.write(f"**Selected month:** {selected_label}")
    st.write(f"**Delay rate 1 month ago:** {lag1_value:.1%}")
    st.write(f"**Delay rate 12 months ago:** {lag12_value:.1%}")
    st.write(
        f"**Delay rate gradient:** {gradient:+.1%} "
        f"({'improving' if gradient < 0 else 'worsening' if gradient > 0 else 'stable'})"
    )
with ctx_col2:
    st.write(f"**Scheduled sectors:** {display_sectors}")
    st.write(f"**Number of public holidays:** {display_holidays:.0f}")
    st.write(f"**Percentage of school holidays:** {display_pct_school:.0%}")

