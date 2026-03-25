"""
Model evaluation dashboard.
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
        width: auto;
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
st.title("Model Evaluation")
st.markdown(
    "In order to assess the quality of the models, a number of evaluation metrics are computed and presented below."
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


@st.cache_data
def get_airline_lookup():
    """Return dict (year_month, route, delay_rate_rounded) -> airline from training CSV."""
    from src.data_loader import PROJECT_ROOT
    csv_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'ml_training_data_multiroute_hols.csv')
    df = pd.read_csv(csv_path, usecols=['year_month', 'departing_port', 'arriving_port', 'airline',
                                        'sectors_flown', 'arrivals_on_time'])
    df['route'] = df['departing_port'] + '_' + df['arriving_port']
    df['delay_rate'] = ((df['sectors_flown'] - df['arrivals_on_time']) / df['sectors_flown']).round(10)
    return df.set_index(['year_month', 'route', 'delay_rate'])['airline'].to_dict()


metadata = get_metadata()
fmetadata = get_forecasting_metadata()
preds = get_test_predictions()
full_preds = get_full_predictions()
fpreds = get_forecasting_test_predictions()
ffull_preds = get_forecasting_full_predictions() if os.path.exists(
    os.path.join(FORECASTING_MODELS_DIR, 'full_predictions.json')
) else None
airline_lookup = get_airline_lookup()

st.divider()

# ---- 1. Summary Table ----
st.subheader("1. Single-Number Evaluation Metrics")

st.markdown("""
            A single-number evaluation metric is a quantitative measure that summarises a model's performance in a single number, typically covering one specific aspect of the performance.  
            Tables 4 to 7 summarise the single-number evaluation metrics computed for all the models considered in this project:

            - **R²**: measures how well the model explains variation in delay rates, where a score of 1.0 means perfect prediction for all data points and 0.0 means no better than guessing the average.  
                For the models presented here, most of the R² values are just above 0.5, which means the models explain slightly more than half of the month-to-month variation in delay rate.
            - **MAE** (Mean Absolute Error) and **RMSE** (Root-Mean-Square Error): both measure prediction accuracy in percentage points.  
                MAE (about 6% for all models) is the average error size, while RMSE (about 8% for all models) penalizes larger errors more due to the squaring operation.  
                Here, RMSE is only about 35% higher than MAE, which indicates the majority of the errors are small.  
            - **Precision**: of all the _predicted_ high-delay months, how often is it correct?
            - **Recall**: of all the _actual_ high-delay months, how many does the model correctly identify?
            - **F1**: a score that captures the balance of precision and recall.
            - **AUC**: when ranking the months by high-delay month probability, how well does the model rank the high-delay months higher than the normal months?  

            All of the single-number evaluation metrics here are computed using the testing data (separate to the training data), to ensure that the model performance is evaluated on unseen data. 
            """)
st.markdown("")
st.space('small')

col_now, col_fore = st.columns(2)

with col_now:
    #st.markdown("##### NOWCASTING")
    st.caption("**Table 4.** Evaluation metrics of the regression models under the **nowcasting** approach.")
    reg_data = []
    for name, key in [("Ridge ★", "ridge"), ("Random Forest", "rf_reg"), ("Neural Network", "nn_reg")]:
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
    st.caption("**Table 5.** Evaluation metrics of the classification models under the **nowcasting** approach.")
    clf_data = []
    for name, key in [("Logistic", "logreg"), ("Random Forest", "rf_clf"), ("XGBoost ★", "xgb_clf"), ("Neural Network", "nn_clf")]:
        if key in metadata['metrics']:
            m = metadata['metrics'][key]
            clf_data.append({
                'Model': name,
                'F1': f"{m['F1']:.4f}",
                'AUC': f"{m['AUC']:.4f}",
                'Precision': f"{m['Precision']:.4f}",
                'Recall': f"{m['Recall']:.4f}",
            })
    render_swiss_table(clf_data, columns=["Model", "F1", "Precision", "Recall", "AUC"])
    st.caption(
        f"Train: {metadata['split']['n_train']} samples ({metadata['split']['train']}) | "
        f"Val: {metadata['split']['n_val']} samples ({metadata['split']['val']}) | "
        f"Test: {metadata['split']['n_test']} samples ({metadata['split']['test']})"
    )

with col_fore:
    #st.markdown("##### FORECASTING")
    st.caption("**Table 6.** Evaluation metrics of the regression models under the **forecasting** approach.")
    freg_data = []
    for name, key in [("Ridge", "ridge"), ("Random Forest", "rf_reg"), ("Neural Network ★", "nn_reg")]:
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
    st.caption("**Table 7.** Evaluation metrics of the classification models under the **forecasting** approach.")
    fclf_data = []
    for name, key in [("Logistic", "logreg"), ("Random Forest", "rf_clf"), ("XGBoost ★", "xgb_clf"), ("Neural Network", "nn_clf")]:
        if key in fmetadata['metrics']:
            m = fmetadata['metrics'][key]
            fclf_data.append({
                'Model': name,
                'F1': f"{m['F1']:.4f}",
                'Precision': f"{m['Precision']:.4f}",
                'Recall': f"{m['Recall']:.4f}",
                'AUC': f"{m['AUC']:.4f}",
            })
    render_swiss_table(fclf_data, columns=["Model", "F1", "Precision", "Recall", "AUC"])
st.divider()

###

# ________________________________
# ---- 2. Baseline Comparison ----
st.subheader("2. Comparison vs Naive Baseline")

st.markdown("""
            A naive baseline provides a benchmark for the minimum performance expected for a specific task.  
            If a trained model cannot outperform this benchmark, the added complexity introduced by the machine learning models cannot be justified.  
            Here, the naive baseline for the regression models is defined as: a model that simply assumes that this month's delay rate stays the same as last month (Lag1 Baseline).  
            The comparisons shown in Figures 3 and 4 show that the machine learning model improves R² by about 0.2 over the naive baseline.  
            This observation suggests that the regression models employed in this project offer meaningful improvements beyond the naive approach.
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
    title="",
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
    title="",
    yaxis=dict(range=[0, max(f_r2_values) * 1.2]),
)
style_plot(fig_f, height=400)

_, col_now, _, col_fore, _ = st.columns([1, 3, 0.3, 3, 1])
with col_now:
    st.plotly_chart(fig, use_container_width=True)
    st.caption("**Figure 3.** R² comparison of the regression nowcasting models vs `lag_1` naive baseline.")

with col_fore:
    st.plotly_chart(fig_f, use_container_width=True)
    f_improvement = f_ridge_r2 - f_baseline_lag1
    st.caption("**Figure 4.** R² comparison of the regression forecasting models vs `lag_1` naive baseline.")

st.divider()

#__________________________________
# ---- 3. Actual vs Prediction ----
st.subheader("3. Actual vs Prediction")

st.markdown("""
            This section visualises how closely the model predictions match the actual delay rates, using time-series for the regression models and confusion matrix for the classification models.  
            """)
st.markdown("""
              
            """)
st.markdown("""
            ##### Time-series

            For the **regression** models, a time-series plot can clearly show how the models perform month-to-month across the years.  
            Figures 5 and 6 compare the time-series of the _predicted_ vs the _actual_ delay rates, under the nowcasting and forecasting approaches, respectively.  
            The delay rates presented here are aggregated monthly across all routes and airlines.

            **Note**: click the legend entries to show/hide individual lines.
            """)

def build_ts_figure(fp):
    """Build a time-series figure from a full_predictions dict. Returns None if unavailable."""
    if fp is None or 'year_month' not in fp:
        return None
    f_ridge = np.array(fp['ridge_pred'])
    f_rf = np.array(fp['rf_pred'])
    f_nn = np.array(fp['nn_reg_pred']) if 'nn_reg_pred' in fp else None
    f_actual = np.array(fp['y_true_reg'])

    df = pd.DataFrame({
        'year_month_dt': pd.to_datetime(fp['year_month']),
        'actual': f_actual,
        'ridge_pred': f_ridge,
        'rf_pred': f_rf,
    })
    if f_nn is not None:
        df['nn_pred'] = f_nn

    agg_dict = {'actual': 'mean', 'ridge_pred': 'mean', 'rf_pred': 'mean'}
    if f_nn is not None:
        agg_dict['nn_pred'] = 'mean'
    agg = df.groupby('year_month_dt').agg(agg_dict).reset_index().sort_values('year_month_dt')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg['year_month_dt'], y=agg['actual'],
        mode='lines+markers', name='Actual',
        line=dict(color=COL["actual"], width=2.2), marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=agg['year_month_dt'], y=agg['ridge_pred'],
        mode='lines+markers', name='Ridge',
        line=dict(color=COL["ridge"], width=1.8, dash='dot'), marker=dict(size=8, symbol='square'),
    ))
    fig.add_trace(go.Scatter(
        x=agg['year_month_dt'], y=agg['rf_pred'],
        mode='lines+markers', name='Random Forest',
        line=dict(color=COL["rf"], width=1.8, dash='dash'), marker=dict(size=6),
    ))
    if f_nn is not None:
        fig.add_trace(go.Scatter(
            x=agg['year_month_dt'], y=agg['nn_pred'],
            mode='lines+markers', name='Neural Network',
            line=dict(color=COL["nn"], width=1.8, dash='dashdot'), marker=dict(size=6, symbol='x'),
        ))
    fig.add_vrect(
        x0="2020-01-01", x1="2022-12-31",
        fillcolor="rgba(13,13,13,0.2)", layer="below", line_width=0,
        annotation_text="COVID-19", annotation_position="top left",
        annotation_font_size=13, annotation_font_color="#505050",
    )
    fig.update_layout(
        title="",
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
    style_plot(fig, height=500)
    return fig


fig_ts_now = build_ts_figure(full_preds)
fig_ts_fore = build_ts_figure(ffull_preds)

if fig_ts_now is not None:
    st.plotly_chart(fig_ts_now, use_container_width=True)
    st.caption("**Figure 5.** Time-series comparisons of the predicted vs actual delay rate under the **nowcasting** approach.")
else:
    st.info("Nowcasting time series chart unavailable.")

st.space("medium")

if fig_ts_fore is not None:
    st.plotly_chart(fig_ts_fore, use_container_width=True)
    st.caption("**Figure 6.** Time-series comparisons of the predicted vs actual delay rate under the **forecasting** approach.")
else:
    st.info("Forecasting time series chart unavailable — run **Update Training** to generate forecasting predictions.")

st.space("medium")

def build_confusion_figure(p):
    """Build a confusion matrix subplot figure from a predictions dict."""
    clf_list = []
    if 'xgb_pred' in p:
        clf_list.append(('XGBoost', np.array(p['xgb_pred'])))
    if 'rf_clf_pred' in p:
        clf_list.append(('Random Forest', np.array(p['rf_clf_pred'])))
    if 'nn_clf_pred' in p:
        clf_list.append(('Neural Network', np.array(p['nn_clf_pred'])))
    if not clf_list:
        return None
    y_true_clf = np.array(p['y_true_clf'])
    n_clf = len(clf_list)
    fig = make_subplots(
        rows=1, cols=n_clf,
        subplot_titles=[name for name, _ in clf_list],
        horizontal_spacing=0.12,
    )
    labels = ['Normal', 'High Delay']
    cm_labels = [['True Negative', 'False Positive'], ['False Negative', 'True Positive']]
    for i, (_, y_pred_clf) in enumerate(clf_list):
        tp = np.sum((y_true_clf == 1) & (y_pred_clf == 1))
        tn = np.sum((y_true_clf == 0) & (y_pred_clf == 0))
        fp = np.sum((y_true_clf == 0) & (y_pred_clf == 1))
        fn = np.sum((y_true_clf == 1) & (y_pred_clf == 0))
        cm = np.array([[tn, fp], [fn, tp]])
        if i == 0:
            cell_text = [[f"{cm_labels[r][c]}<br>{cm[r][c]}" for c in range(2)] for r in range(2)]
        else:
            cell_text = [[str(cm[r][c]) for c in range(2)] for r in range(2)]
        fig.add_trace(go.Heatmap(
            z=cm, x=labels, y=labels,
            text=cell_text,
            texttemplate='%{text}',
            textfont={'size': 13},
            colorscale=[[0.0, "#f8f7f3"], [1.0, "#1f4e79"]],
            showscale=False,
        ), row=1, col=i+1)
        fig.update_xaxes(title_text='Predicted', row=1, col=i+1)
        fig.update_yaxes(title_text='Actual', row=1, col=i+1)
    fig.update_layout(title="")
    style_plot(fig, height=350)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig

fig_cm_now = build_confusion_figure(preds)
fig_cm_fore = build_confusion_figure(fpreds)

st.markdown("""
            ##### Confusion matrix

            For the **classification** models, a confusion matrix summarises a model's performance by comparing its predicted classes against the actual classes.  
            Figures 7 and 8 show the confusion matrices of the classification models employed to predict high-delay months, under the nowcasting and forecasting approaches, respectively.  
            Logistic regression model is not included here since it displays similar behaviour to Random Forest, but with worse performance.

            The results here echo the observations made from the single-value evaluation metrics above:
            - XGBoost has the best balance in minimising False Positives and False Negatives.
            - Random Forest has a stronger tendency to predict False Negative, i.e. predicting a normal month when it is actually a high-delay month (low recall).
            - Neural Network has a stronger tendency to predict False Positive, i.e. predicting a high-delay month when it is actually a normal month (low precision).
            """)

if fig_cm_now is not None:
    st.plotly_chart(fig_cm_now, use_container_width=True)
    st.caption("**Figure 7.** Confusion matrices for the classification models under the **nowcasting** approach.")

st.space("small")

if fig_cm_fore is not None:
    st.plotly_chart(fig_cm_fore, use_container_width=True)
    st.caption("**Figure 8.** Confusion matrices for the classification models under the **forecasting** approach.")


# ____________________________________
# ---- 4. Route-Level Performance ----
st.divider()
st.subheader("4. Route-Level Performance")

st.markdown("""
            The prediction accuracy of the models varies depending on the route.

            There are two factors affecting the route-specific accuracy:
            - inherent volatility: some routes exhibit greater delay rate volatility throughout the historical record, reducing the predictability of the underlying signal;
            - flight volume: routes with fewer than 50 flights per month exhibit high delay rate volatility, where a single delayed flight can shift the monthly rate by several percentage points.  
            
            Figure 9 compares the R² values of the predictions made for each route under the nowcasting (Ridge) and the forecasting (Neural Network) approaches.  
            
            """)

# Build forecasting route DataFrame, sorted by Ridge R² (used as sort order)
froute_data = []
for route, m in fmetadata['route_metrics'].items():
    froute_data.append({
        'Route': route.replace('_', ' \u2192 '),
        'Ridge R²': m['ridge_r2'],
        'RF R²': m['rf_r2'],
        'Lag1 R²': m['lag1_r2'],
    })
froute_df = pd.DataFrame(froute_data).sort_values('Ridge R²', ascending=True)

# Build nowcasting route DataFrame, aligned to forecasting sort order
route_data = []
for route, m in metadata['route_metrics'].items():
    route_data.append({
        'Route': route.replace('_', ' \u2192 '),
        'Ridge R²': m['ridge_r2'],
        'RF R²': m['rf_r2'],
        'Lag1 R²': m['lag1_r2'],
    })
route_df = pd.DataFrame(route_data)
route_df = route_df.set_index('Route').reindex(froute_df['Route']).reset_index()

fig_route = go.Figure()
fig_route.add_trace(go.Bar(
    y=froute_df['Route'], x=froute_df['Ridge R²'],
    orientation='h', name='Forecasting',
    marker_color='#ff7f00',
    offsetgroup=1,
))
fig_route.add_trace(go.Bar(
    y=route_df['Route'], x=route_df['Ridge R²'],
    orientation='h', name='Nowcasting',
    marker_color=COL["ridge"],
    offsetgroup=0,
))
fig_route.update_layout(
    barmode='group',
    bargap=0.35,
    bargroupgap=0.08,
    title="",
    xaxis_title="R²",
    xaxis=dict(range=[min(route_df['Ridge R²'].min(), froute_df['Ridge R²'].min()) - 0.05, 0.85]),
)
style_plot(fig_route, height=max(420, len(route_df) * 40))
col_chart, _ = st.columns([3, 1])
with col_chart:
    st.plotly_chart(fig_route, use_container_width=True)
st.caption("**Figure 9.** Comparison of delay rate prediction accuracy (R²) by route, comparing nowcasting (Ridge) and forecasting (Neural Network) approaches.")

st.divider()

# ___________________________________
# ---- 5. Error Analysis ----
st.subheader("5. Error Analysis")

st.markdown("""
            This section investigates the residuals (prediction errors) of the forecasting Neural Network model on the test dataset: examining the residual histogram distribution and the worst-predicted routes.  
            Residuals are defined as (_predicted delay rate_ − _actual delay rate_), such that a posive residual value indicates that the model is overpredicting the delay rate and vice versa.
            """)

# Build error DataFrame from forecasting test predictions
_f_y_true = np.array(fpreds['y_true_reg'])
_f_nn_pred = np.array(fpreds['nn_reg_pred'])
_f_residuals = _f_nn_pred - _f_y_true
_f_abs_errors = np.abs(_f_residuals)

# Recover year_month by matching (y_true, ridge_pred) pairs against full_predictions
_ffull_ym_lookup = {
    (round(y, 10), round(r, 10)): ym
    for ym, s, y, r in zip(ffull_preds['year_month'], ffull_preds['split'], ffull_preds['y_true_reg'], ffull_preds['ridge_pred'])
    if s == 'test'
} if ffull_preds is not None else {}
_f_year_months = [
    _ffull_ym_lookup.get((round(y, 10), round(r, 10)), '')
    for y, r in zip(fpreds['y_true_reg'], fpreds['ridge_pred'])
]

_f_airlines = [
    airline_lookup.get((ym, route, round(y, 10)), '')
    for ym, route, y in zip(_f_year_months, fpreds['routes'], fpreds['y_true_reg'])
]

ferr_df = pd.DataFrame({
    'year_month': _f_year_months,
    'route': [r.replace('_', ' → ') for r in fpreds['routes']],
    'airline': _f_airlines,
    'actual': _f_y_true,
    'predicted': _f_nn_pred,
    'residual': _f_residuals,
    'abs_error': _f_abs_errors,
})

_f_mean_residual = float(_f_residuals.mean())
_f_bias_dir = "over-prediction" if _f_mean_residual > 0 else "under-prediction"

# ---- 5a. Residual distribution ----
st.markdown("""
            #### Residual histogram

            Figure 10 shows the histogram of the residual, where the distribution is roughly symmetric around zero, with no pronounced tail in either direction.  
            This indicates that the prediction model is well-calibrated, as a non-zero mean indicates a systematic bias is present.
            """)

fig_resid = go.Figure()
fig_resid.add_trace(go.Histogram(
    x=_f_residuals,
    nbinsx=40,
    marker_color="#0E0E0E",
    opacity=0.85,
    name='Residuals',
    showlegend=False,
))
_resid_y_max = 450
fig_resid.add_trace(go.Scatter(
    x=[_f_mean_residual, _f_mean_residual], y=[0, _resid_y_max],
    mode='lines', line=dict(color=COL["negative"] if _f_mean_residual > 0 else COL["positive"], width=3, dash="dot"),
    name=f'Mean = {_f_mean_residual:+.3f}', showlegend=True,
))
fig_resid.update_layout(xaxis_title="Residual", yaxis_title="Count", title=" ")
style_plot(fig_resid, height=380)
col_chart, _ = st.columns([4, 3])
with col_chart:
    st.plotly_chart(fig_resid, use_container_width=True)
st.caption(
    f"**Figure 10.** Histogram of the prediction residual for the forecasting Neural Network model on the test set."
)

st.space("small")

# ---- 5b. Worst-predictions----
st.markdown("""
            #### Highest prediction errors

            Table 8 below lists 10 route-month combinations where the forecasting Neural Network model made the largest absolute errors (expressed in percentage point, pp) on the test dataset.  
            Such sampling reveals the cases where the model's assumptions break down, such as sudden operational changes or even potentially inconsistent reporting.

            For example, the top 2 entries with 100% delay rate occured when QantasLink only have 1 reported flight each month for the Adelaide to Brisbane route (according to BITRE report).  
            This was either an operational anomaly or a reporting error, since QantasLink operated 45 flights for the same route in March 2019, just 2 months prior.

            Another example may be examined from the repeated delay rate underprediction in November 2025.  
            These were caused by the unexpectedly high delay rate due to a combination of [air traffic communication outage](https://australianaviation.com.au/2025/11/airservices-resolves-comms-outage-that-delayed-flights) and [grounded A320 airplanes](https://www.smh.com.au/national/global-software-outage-triggers-australian-flight-delays-20251129-p5njew.html). 
            """)

worst_rows = [
    {
        'Year-Month': row['year_month'],
        'Route': row['route'],
        'Airline': row['airline'],
        'Actual (%)': f"{row['actual'] * 100:.1f}",
        'Predicted (%)': f"{row['predicted'] * 100:.1f}",
        'Absolute error (pp)': f"{row['abs_error'] * 100:.1f}",
    }
    for _, row in ferr_df.nlargest(10, 'abs_error').iterrows()
]
st.caption("**Table 8.** Top 10 highest absolute prediction errors made by the forecasting Neural Network model on test set).")
render_swiss_table(worst_rows, ['Year-Month', 'Route', 'Airline', 'Actual (%)', 'Predicted (%)', 'Absolute error (pp)'])


