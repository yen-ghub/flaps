"""
Model performance dashboard.
Displays evaluation metrics, baseline comparisons, and route-level analysis.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import json
from html import escape

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.data_loader import NOWCASTING_MODELS_DIR, FORECASTING_MODELS_DIR, load_metadata, load_forecasting_metadata
from src.ui_theme import apply_theme

st.set_page_config(page_title="FLAPS — Model Performance", page_icon="✈️", layout="wide")
apply_theme(
    extra_css="""
    .swiss-note {
        color: #5a5a5a;
        line-height: 1.5;
        margin-bottom: 1rem;
    }
    .swiss-table-wrap {
        border: 0;
        border-radius: 0;
        overflow: hidden;
        margin: 0;
        padding: 0;
    }
    .swiss-table {
        width: 100%;
        border-collapse: collapse;
        border: 1px solid rgba(13, 13, 13, 0.25);
    }
    .swiss-table th {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.72rem;
        font-weight: 600;
        text-align: left;
        line-height: 1.2;
        padding: 0.48rem 0.62rem;
        border-bottom: 1px solid rgba(13, 13, 13, 0.25);
        background: #f8f7f3;
    }
    .swiss-table td {
        line-height: 1.2;
        padding: 0.46rem 0.62rem;
        border-bottom: 1px solid rgba(13, 13, 13, 0.25);
    }
    .swiss-table th + th,
    .swiss-table td + td {
        border-left: 1px solid rgba(13, 13, 13, 0.25);
    }
    .swiss-table tbody tr:last-child td {
        border-bottom: none;
    }
    """
)
st.title("Model Performance")
st.markdown(
    "Consolidated evaluation dashboard for nowcasting and forecasting performance, including baseline benchmarking, route-level diagnostics, and prediction-behaviour visualisations."
)

COL = {
    "actual": "#0d0d0d",
    "ridge": "#377eb8",              # ColorBrewer Set1 blue
    "rf": "#4daf4a",                 # ColorBrewer Set1 green
    "nn": "#984ea3",                 # ColorBrewer Set1 purple
    "xgb": "#8b3f2f",
    "baseline_mean": "#9a9a9a",
    "baseline_lag1": "#666666",
    "positive": "#2f6b3c",
    "negative": "#a13d3d",
    "grid": "rgba(13, 13, 13, 0.12)",
    "border": "rgba(13, 13, 13, 0.25)",
    "bg": "#efeee9",
}


def style_plot(fig, *, height=420):
    """Apply consistent Swiss-style Plotly formatting."""
    fig.update_layout(
        template="simple_white",
        paper_bgcolor=COL["bg"],
        plot_bgcolor=COL["bg"],
        font=dict(color=COL["actual"], size=16),
        title_font=dict(size=20),
        margin=dict(l=20, r=20, t=64, b=28),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=16)),
    )
    fig.update_xaxes(
        showline=True,
        linecolor=COL["border"],
        linewidth=1,
        showgrid=True,
        gridcolor=COL["grid"],
        zeroline=False,
        ticks="outside",
        tickfont=dict(size=14),
        title_font=dict(size=15),
    )
    fig.update_yaxes(
        showline=True,
        linecolor=COL["border"],
        linewidth=1,
        showgrid=True,
        gridcolor=COL["grid"],
        zeroline=False,
        ticks="outside",
        tickfont=dict(size=14),
        title_font=dict(size=15),
    )


def render_swiss_table(rows, columns):
    """Render compact Swiss-style HTML table."""
    if not rows:
        return
    header_html = "".join(f"<th>{escape(str(col))}</th>" for col in columns)
    body_html = "".join(
        "<tr>" + "".join(
            f"<td>{escape(' '.join(str(row.get(col, '')).split()))}</td>" for col in columns
        ) + "</tr>"
        for row in rows
    )
    st.markdown(
        f"""
        <div class="swiss-table-wrap">
            <table class="swiss-table">
                <thead><tr>{header_html}</tr></thead>
                <tbody>{body_html}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def get_metadata():
    return load_metadata()


@st.cache_data
def get_forecasting_metadata():
    return load_forecasting_metadata()


@st.cache_data
def get_test_predictions():
    with open(os.path.join(NOWCASTING_MODELS_DIR, 'test_predictions.json'), 'r') as f:
        return json.load(f)


@st.cache_data
def get_full_predictions():
    with open(os.path.join(NOWCASTING_MODELS_DIR, 'full_predictions.json'), 'r') as f:
        return json.load(f)


@st.cache_data
def get_forecasting_test_predictions():
    with open(os.path.join(FORECASTING_MODELS_DIR, 'test_predictions.json'), 'r') as f:
        return json.load(f)


@st.cache_data
def get_forecasting_full_predictions():
    with open(os.path.join(FORECASTING_MODELS_DIR, 'full_predictions.json'), 'r') as f:
        return json.load(f)


metadata = get_metadata()
fmetadata = get_forecasting_metadata()
preds = get_test_predictions()
full_preds = get_full_predictions()
fpreds = get_forecasting_test_predictions()
ffull_preds = get_forecasting_full_predictions() if os.path.exists(
    os.path.join(FORECASTING_MODELS_DIR, 'full_predictions.json')
) else None

st.divider()

# ---- 1. Summary Table ----
st.subheader("1. Performance Metrics Summary")

st.markdown("**R²**: measures how well the model explains variation in delay rates: a score of 1.0 is perfect, 0.0 means no better than guessing the average.  \n"
            "For the models presented here, R² is about 0.5, which means the models explain roughly half the month-to-month variation in delays."
            )

st.markdown("**MAE** (Mean Absolute Error) and **RMSE** (Root-Mean-Square Error): both measure prediction accuracy in percentage points.  \n"
            "MAE (about 6%) is the average error size, while RMSE (about 8%) penalizes larger errors more.  \n"
            "Here, RMSE is about 35%  higher than MAE, which indicates the majority of the errors are small.   \n"
            )
st.markdown("**Precision**: of all the _predicted_ high-delay months, how often is it correct?  \n"
            "**Recall**: of all the _actual_ high-delay months, how many does the model correctly identify?  \n"
            "**F1**: a score that captures the balance of precision and recall.  \n"
            )
st.markdown("**AUC**: how well does the model rank high-delay months higher than normal-delay months?")
st.space('small')

col_now, col_fore = st.columns(2)

with col_now:
    st.markdown("##### NOWCASTING")
    st.markdown("**Regression (Test Set)**")
    reg_data = []
    for name, key in [("Ridge", "ridge"), ("Random Forest", "rf_reg"), ("Neural Network", "nn_reg")]:
        if key in metadata['metrics']:
            m = metadata['metrics'][key]
            reg_data.append({
                'Model': name,
                'R²': f"{m['R2']:.4f}",
                'RMSE': f"{m['RMSE']:.4f}",
                'MAE': f"{m['MAE']:.4f}",
            })
    render_swiss_table(reg_data, columns=["Model", "R²", "RMSE", "MAE"])
    st.space('small')
    st.markdown("**Classification (Test Set)**")
    clf_data = []
    for name, key in [("Logistic", "logreg"), ("Random Forest", "rf_clf"), ("XGBoost", "xgb_clf"), ("Neural Network", "nn_clf")]:
        if key in metadata['metrics']:
            m = metadata['metrics'][key]
            clf_data.append({
                'Model': name,
                'F1': f"{m['F1']:.4f}",
                'AUC': f"{m['AUC']:.4f}",
                'Precision': f"{m['Precision']:.4f}",
                'Recall': f"{m['Recall']:.4f}",
            })
    render_swiss_table(clf_data, columns=["Model", "F1", "AUC", "Precision", "Recall"])
    st.caption(
        f"Train: {metadata['split']['n_train']} samples ({metadata['split']['train']}) | "
        f"Val: {metadata['split']['n_val']} samples ({metadata['split']['val']}) | "
        f"Test: {metadata['split']['n_test']} samples ({metadata['split']['test']})"
    )

with col_fore:
    st.markdown("##### FORECASTING")
    st.markdown("**Regression (Test Set)**")
    freg_data = []
    for name, key in [("Ridge", "ridge"), ("Random Forest", "rf_reg"), ("Neural Network", "nn_reg")]:
        if key in fmetadata['metrics']:
            m = fmetadata['metrics'][key]
            freg_data.append({
                'Model': name,
                'R²': f"{m['R2']:.4f}",
                'RMSE': f"{m['RMSE']:.4f}",
                'MAE': f"{m['MAE']:.4f}",
            })
    render_swiss_table(freg_data, columns=["Model", "R²", "RMSE", "MAE"])
    st.space('small')
    st.markdown("**Classification (Test Set)**")
    fclf_data = []
    for name, key in [("Logistic", "logreg"), ("Random Forest", "rf_clf"), ("XGBoost", "xgb_clf"), ("Neural Network", "nn_clf")]:
        if key in fmetadata['metrics']:
            m = fmetadata['metrics'][key]
            fclf_data.append({
                'Model': name,
                'F1': f"{m['F1']:.4f}",
                'AUC': f"{m['AUC']:.4f}",
                'Precision': f"{m['Precision']:.4f}",
                'Recall': f"{m['Recall']:.4f}",
            })
    render_swiss_table(fclf_data, columns=["Model", "F1", "AUC", "Precision", "Recall"])
    # st.caption(
    #     f"Train: {fmetadata['split']['n_train']} samples ({fmetadata['split']['train']}) | "
    #     f"Val: {fmetadata['split']['n_val']} samples ({fmetadata['split']['val']}) | "
    #     f"Test: {fmetadata['split']['n_test']} samples ({fmetadata['split']['test']})"
    # )


st.divider()


# ________________________________
# ---- 2. Baseline Comparison ----
st.subheader("2. Comparison vs Naive Baseline")

st.markdown("""
            A naive baseline provides a benchmark for the minimum performance expected for this task.

            If a trained model cannot outperform this benchmark, the added complexity cannot be justified.

            Here, the naive baseline is defined as: assuming the the next month's delay is equal the same as this month's (_Lag1 Baseline_).
            
            The results suggest the machine learning models offer meaningful improvements beyond the naive approach.
            """)

# Nowcasting
baseline_lag1 = metadata['baseline_lag1_r2']
ridge_r2 = metadata['metrics']['ridge']['R2']
rf_r2 = metadata['metrics']['rf_reg']['R2']
nn_r2 = metadata['metrics'].get('nn_reg', {}).get('R2')

#st.markdown("**Nowcasting Models**")
fig = go.Figure()
models_names = ['Lag1 Baseline', 'Ridge', 'Random Forest']
r2_values = [baseline_lag1, ridge_r2, rf_r2]
bar_colors = [COL["baseline_lag1"], COL["ridge"], COL["rf"]]

if nn_r2 is not None:
    models_names.append('Neural Network')
    r2_values.append(nn_r2)
    bar_colors.append(COL["nn"])

fig.add_trace(go.Bar(
    x=models_names, y=r2_values,
    marker_color=bar_colors,
    text=[f"{v:.3f}" for v in r2_values],
    textposition='outside',
))
fig.update_layout(
    yaxis_title="R²",
    #title="NOWCASTING",
    yaxis=dict(range=[0, max(r2_values) * 1.2]),
)
style_plot(fig, height=400)

# Forecasting
f_baseline_lag1 = fmetadata['baseline_lag1_r2']
f_ridge_r2 = fmetadata['metrics']['ridge']['R2']
f_rf_r2 = fmetadata['metrics']['rf_reg']['R2']
f_nn_r2 = fmetadata['metrics'].get('nn_reg', {}).get('R2')

fig_f = go.Figure()
f_models_names = ['Lag1 Baseline', 'Ridge', 'Random Forest']
f_r2_values = [f_baseline_lag1, f_ridge_r2, f_rf_r2]
f_bar_colors = [COL["baseline_lag1"], COL["ridge"], COL["rf"]]

if f_nn_r2 is not None:
    f_models_names.append('Neural Network')
    f_r2_values.append(f_nn_r2)
    f_bar_colors.append(COL["nn"])

fig_f.add_trace(go.Bar(
    x=f_models_names, y=f_r2_values,
    marker_color=f_bar_colors,
    text=[f"{v:.3f}" for v in f_r2_values],
    textposition='outside',
))
fig_f.update_layout(
    yaxis_title="R²",
    title="FORECASTING",
    yaxis=dict(range=[0, max(f_r2_values) * 1.2]),
)
style_plot(fig_f, height=400)

col_now, col_fore = st.columns(2)
with col_now:
    st.plotly_chart(fig, use_container_width=True)
    improvement = ridge_r2 - baseline_lag1
    # st.markdown(
    #     f"The lag1 baseline achieves R² = {baseline_lag1:.3f}. "
    #     f"The Ridge model improves by **+{improvement:.3f}**, demonstrating that weather, "
    #     f"holidays, momentum, and seasonal encoding add predictive value beyond trivial persistence."
    # )
with col_fore:
    st.plotly_chart(fig_f, use_container_width=True)
    f_improvement = f_ridge_r2 - f_baseline_lag1
    # st.markdown(
    #     f"The lag1 baseline achieves R² = {f_baseline_lag1:.3f}. "
    #     f"The Ridge model improves by **+{f_improvement:.3f}**, demonstrating that lagged features, "
    #     f"holidays, momentum, and seasonal encoding add predictive value beyond trivial persistence."
    # )

st.divider()

# ____________________________________
# ---- 3. Route-Level Performance ----
st.subheader("3. Route-Level Performance")

st.markdown("""
            The predictive performance of the models varies depending on the route.
            
            The chart below breaks down the accuracy of the Ridge model for each flight route.
            """)

# Build nowcasting route DataFrame, sorted by Ridge R²
route_data = []
for route, m in metadata['route_metrics'].items():
    route_data.append({
        'Route': route.replace('_', ' \u2192 '),
        'Ridge R²': m['ridge_r2'],
        'RF R²': m['rf_r2'],
        'Lag1 R²': m['lag1_r2'],
    })
route_df = pd.DataFrame(route_data).sort_values('Ridge R²', ascending=True)

# Build forecasting route DataFrame, aligned to same route order
froute_data = []
for route, m in fmetadata['route_metrics'].items():
    froute_data.append({
        'Route': route.replace('_', ' \u2192 '),
        'Ridge R²': m['ridge_r2'],
        'RF R²': m['rf_r2'],
        'Lag1 R²': m['lag1_r2'],
    })
froute_df = pd.DataFrame(froute_data)
froute_df = froute_df.set_index('Route').reindex(route_df['Route']).reset_index()

fig = go.Figure()
fig.add_trace(go.Bar(
    y=route_df['Route'], x=route_df['Ridge R²'],
    orientation='h', name='NOWCASTING',
    marker_color=COL["ridge"],
))
fig.add_trace(go.Bar(
    y=froute_df['Route'], x=froute_df['Ridge R²'],
    orientation='h', name='FORECASTING',
    marker_color='#ff7f00',
))
fig.update_layout(
    barmode='group',
    bargap=0.35,
    bargroupgap=0.08,
    title="Ridge R² by Route",
    xaxis_title="R²",
)
style_plot(fig, height=600)
col_chart, _ = st.columns([2, 1])
with col_chart:
    st.plotly_chart(fig, use_container_width=True)

st.divider()

#__________________________________
# ---- 4. Actual vs Prediction ----
st.subheader("4. Actual vs Prediction")

avp_view = st.radio(
    "Model type", ["NOWCASTING", "FORECASTING"],
    horizontal=True, label_visibility="collapsed",
    key="avp_view",
)

_is_nowcast = avp_view == "NOWCASTING"
_preds = preds if _is_nowcast else fpreds
_full_preds = full_preds if _is_nowcast else ffull_preds
_meta = metadata if _is_nowcast else fmetadata
_ridge_color = COL["ridge"]

y_true = np.array(_preds['y_true_reg'])
ridge_preds = np.array(_preds['ridge_pred'])
rf_preds = np.array(_preds['rf_pred'])
nn_preds = np.array(_preds['nn_reg_pred']) if 'nn_reg_pred' in _preds else None

_ridge_r2 = _meta['metrics']['ridge']['R2']
_rf_r2 = _meta['metrics']['rf_reg']['R2']
_nn_r2 = _meta['metrics'].get('nn_reg', {}).get('R2')

# Time series visualization (full date range)
if _full_preds is not None and 'year_month' in _full_preds:
    # st.markdown("**Time Series**  \n"
    #             "Click the legend to show/hide individual lines.")

    full_ridge = np.array(_full_preds['ridge_pred'])
    full_rf = np.array(_full_preds['rf_pred'])
    full_nn = np.array(_full_preds['nn_reg_pred']) if 'nn_reg_pred' in _full_preds else None
    full_actual = np.array(_full_preds['y_true_reg'])

    ts_df = pd.DataFrame({
        'year_month': _full_preds['year_month'],
        'actual': full_actual,
        'ridge_pred': full_ridge,
        'rf_pred': full_rf,
        'split': _full_preds['split'],
    })
    if full_nn is not None:
        ts_df['nn_pred'] = full_nn

    ts_df['year_month_dt'] = pd.to_datetime(ts_df['year_month'])
    agg_dict = {'actual': 'mean', 'ridge_pred': 'mean', 'rf_pred': 'mean'}
    if full_nn is not None:
        agg_dict['nn_pred'] = 'mean'

    monthly_agg = ts_df.groupby('year_month_dt').agg(agg_dict).reset_index()
    monthly_agg = monthly_agg.sort_values('year_month_dt')

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=monthly_agg['year_month_dt'], y=monthly_agg['actual'],
        mode='lines+markers', name='Actual',
        line=dict(color=COL["actual"], width=2.2), marker=dict(size=8),
    ))
    fig_ts.add_trace(go.Scatter(
        x=monthly_agg['year_month_dt'], y=monthly_agg['ridge_pred'],
        mode='lines+markers', name='Ridge',
        line=dict(color=_ridge_color, width=1.8, dash='dot'), marker=dict(size=8, symbol='square'),
    ))
    fig_ts.add_trace(go.Scatter(
        x=monthly_agg['year_month_dt'], y=monthly_agg['rf_pred'],
        mode='lines+markers', name='Random Forest',
        line=dict(color=COL["rf"], width=1.8, dash='dash'), marker=dict(size=6),
    ))
    if full_nn is not None:
        fig_ts.add_trace(go.Scatter(
            x=monthly_agg['year_month_dt'], y=monthly_agg['nn_pred'],
            mode='lines+markers', name='Neural Network',
            line=dict(color=COL["nn"], width=1.8, dash='dashdot'), marker=dict(size=6, symbol='x'),
        ))
    fig_ts.update_layout(
        title="Time-Series",
        xaxis_title="Month",
        yaxis_title="Delay Rate (Monthly Average)",
        hovermode='x unified',
        legend=dict(
            orientation="v",
            xanchor="left", x=0.01,
            yanchor="top", y=0.99,
            bgcolor="#efeee9",
            bordercolor=COL["border"], borderwidth=1,
            itemclick="toggle", itemdoubleclick="toggleothers",
        ),
    )
    style_plot(fig_ts, height=500)
    st.plotly_chart(fig_ts, use_container_width=True)
    st.caption("Aggregated across all routes and airlines. Click the legend to show/hide individual lines.")


elif _full_preds is None:
    st.info("Time series chart unavailable — run **Update Training** to generate forecasting predictions.")

st.space("medium")
#st.markdown("**Correlations**")

# # Plot actual vs prediction correlations
# n_cols = 3 if nn_preds is not None else 2
# subplot_titles = [f"Ridge (R² = {_ridge_r2:.3f})", f"Random Forest (R² = {_rf_r2:.3f})"]
# if nn_preds is not None:
#     subplot_titles.append(f"Neural Network (R² = {_nn_r2:.3f})")

# fig = make_subplots(rows=1, cols=n_cols, subplot_titles=subplot_titles)
# fig.add_trace(go.Scatter(
#     x=y_true, y=ridge_preds, mode='markers',
#     marker=dict(size=4, opacity=0.45, color=_ridge_color),
#     name='Ridge', showlegend=False,
# ), row=1, col=1)
# fig.add_trace(go.Scatter(
#     x=y_true, y=rf_preds, mode='markers',
#     marker=dict(size=4, opacity=0.45, color=COL["rf"]),
#     name='RF', showlegend=False,
# ), row=1, col=2)
# if nn_preds is not None:
#     fig.add_trace(go.Scatter(
#         x=y_true, y=nn_preds, mode='markers',
#         marker=dict(size=4, opacity=0.45, color=COL["nn"]),
#         name='NN', showlegend=False,
#     ), row=1, col=3)
# for col in range(1, n_cols + 1):
#     fig.add_trace(go.Scatter(
#         x=[0, 0.6], y=[0, 0.6], mode='lines',
#         line=dict(color=COL["baseline_lag1"], dash='dash'), showlegend=False,
#     ), row=1, col=col)
#     fig.update_xaxes(title_text="Actual Delay Rate", range=[0, 0.6], row=1, col=col)
#     fig.update_yaxes(title_text="Predicted Delay Rate", range=[0, 0.6], row=1, col=col)
# fig.update_layout(
#     title="Correlations"
# )
# style_plot(fig, height=450)
# st.plotly_chart(fig, use_container_width=True)

# st.space('medium')

# Classification Confusion Matrices
#st.markdown("**Classification: Confusion Matrices**")
y_true_clf = np.array(_preds['y_true_clf'])
clf_list = []
if 'xgb_pred' in _preds:
    clf_list.append(('XGBoost', np.array(_preds['xgb_pred']), COL["xgb"]))
if 'rf_clf_pred' in _preds:
    clf_list.append(('Random Forest', np.array(_preds['rf_clf_pred']), COL["rf"]))
if 'nn_clf_pred' in _preds:
    clf_list.append(('Neural Network', np.array(_preds['nn_clf_pred']), COL["nn"]))

if clf_list:
    n_clf = len(clf_list)
    fig_cm = make_subplots(
        rows=1, cols=n_clf,
        subplot_titles=[name for name, _, _ in clf_list],
        horizontal_spacing=0.12,
    )
    for i, (name, y_pred_clf, color) in enumerate(clf_list):
        tp = np.sum((y_true_clf == 1) & (y_pred_clf == 1))
        tn = np.sum((y_true_clf == 0) & (y_pred_clf == 0))
        fp = np.sum((y_true_clf == 0) & (y_pred_clf == 1))
        fn = np.sum((y_true_clf == 1) & (y_pred_clf == 0))
        cm = np.array([[tn, fp], [fn, tp]])
        labels = ['Normal', 'High Delay']
        fig_cm.add_trace(go.Heatmap(
            z=cm, x=labels, y=labels,
            text=cm, texttemplate='%{text}', textfont={'size': 14},
            colorscale=[[0.0, "#f8f7f3"], [1.0, "#1f4e79"]],
            showscale=False,
        ), row=1, col=i+1)
        fig_cm.update_xaxes(title_text='Predicted', row=1, col=i+1)
        fig_cm.update_yaxes(title_text='Actual', row=1, col=i+1)
    fig_cm.update_layout(
        title="Confusion Matrices"
    )
    style_plot(fig_cm, height=350)
    fig_cm.update_xaxes(showgrid=False)
    fig_cm.update_yaxes(showgrid=False)
    st.plotly_chart(fig_cm, use_container_width=True)

