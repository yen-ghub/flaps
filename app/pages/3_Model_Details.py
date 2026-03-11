"""
Model architecture dashboard.
Visualises architecture for Ridge, XGBoost, and Neural Network models.
"""

import json
import os
from html import escape

import joblib
import streamlit as st
from src.ui_theme import apply_theme


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NOWCASTING_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "nowcasting")
FORECASTING_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "forecasting")


st.set_page_config(page_title="FLAPS — Model Details", page_icon="✈️", layout="wide")
apply_theme(
    extra_css="""
    .swiss-note {
        max-width: 920px;
    }
    [data-testid="stMetric"] {
        min-height: 120px;
    }
    .arch-flow {
        display: flex;
        flex-direction: row;
        align-items: center;
        flex-wrap: nowrap;
    }
    .arch-node {
        border: 1px solid rgba(13, 13, 13, 0.25);
        background: #f8f7f3;
        min-height: 74px;
        min-width: 120px;
        padding: 0.7rem 0.9rem;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        line-height: 1.3;
        font-size: 0.88rem;
        flex: 0 0 auto;
    }
    .arch-arrow {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0 0.6rem;
        font-size: 1.1rem;
        color: #5a5a5a;
        flex: 0 0 auto;
    }
    .pipeline-gap {
        height: 0.9rem;
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
        table-layout: fixed;
        border-collapse: collapse;
        margin: 0;
        border: 1px solid rgba(13, 13, 13, 0.25);
    }
    .swiss-table th {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.72rem;
        font-weight: 600;
        text-align: left;
        box-sizing: border-box;
        height: 40px;
        line-height: 1.2;
        padding: 0.48rem 0.62rem;
        border-bottom: 1px solid rgba(13, 13, 13, 0.25);
        background: #f8f7f3;
        vertical-align: middle;
    }
    .swiss-table tbody tr {
        height: 40px;
    }
    .swiss-table td {
        box-sizing: border-box;
        line-height: 1.2;
        padding: 0.46rem 0.62rem;
        border-bottom: 1px solid rgba(13, 13, 13, 0.25);
        vertical-align: middle;
    }
    .swiss-table th + th,
    .swiss-table td + td {
        border-left: 1px solid rgba(13, 13, 13, 0.25);
    }
    .swiss-table tbody tr:last-child td {
        border-bottom: none;
    }
    .swiss-table th:nth-child(1), .swiss-table td:nth-child(1) { width: 52px; }
    .swiss-table th:nth-child(2), .swiss-table td:nth-child(2) { width: 220px; }
    .swiss-table th:nth-child(3), .swiss-table td:nth-child(3) { width: 480px; white-space: normal; }
    .swiss-table th:last-child {
        text-align: center;
        width: 420px;
    }
    .swiss-table td:last-child { width: 420px; }
    /* Tight 2-column table (Parameter/Value) */
    .swiss-table-tight th:nth-child(1), .swiss-table-tight td:nth-child(1) { width: 140px; }
    .swiss-table-tight th:nth-child(2), .swiss-table-tight td:nth-child(2) { width: 120px; }
    """
)
st.title("Model Details")
st.markdown("""
            The top 3 machine learning models for the nowcasting and forecasting approaches, when considering both the performance metrics and interpretability, are discussed in further detail below.
            """)

COL = {
    "positive": "#2f6b3c",
    "negative": "#a13d3d",
    "grid": "rgba(13, 13, 13, 0.12)",
    "border": "rgba(13, 13, 13, 0.25)",
    "bg": "#efeee9",
}


@st.cache_data
def load_metadata(model_set):
    path = os.path.join(
        NOWCASTING_MODELS_DIR if model_set == "nowcasting" else FORECASTING_MODELS_DIR,
        "metadata.json",
    )
    with open(path, "r") as f:
        return json.load(f)


@st.cache_resource
def load_joblib_model(path):
    return joblib.load(path)


@st.cache_resource
def load_keras_model(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)


def render_flow_row(nodes):
    parts = []
    for idx, node in enumerate(nodes):
        parts.append(f'<div class="arch-node">{node}</div>')
        if idx < len(nodes) - 1:
            parts.append('<div class="arch-arrow">→</div>')
    inner = "\n".join(parts)
    st.markdown(
        f'<div class="arch-flow">{inner}</div>',
        unsafe_allow_html=True,
    )


def format_float(value, digits=4):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def get_layer_rows(model):
    rows = []
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        units = getattr(layer, "units", None)
        rate = getattr(layer, "rate", None)
        activation = getattr(layer, "activation", None)
        activation_name = activation.__name__ if activation is not None else "-"
        rows.append(
            {
                "Layer": layer.name,
                "Type": layer_type,
                "Units": int(units) if units is not None else "-",
                "Activation": activation_name,
                "Dropout": format_float(rate, 2) if rate is not None else "-",
                "Params": int(layer.count_params()),
            }
        )
    return rows


def render_swiss_table(rows, columns, table_class="swiss-table"):
    """Render a compact Swiss-style HTML table."""
    if not rows:
        return
    header_html = "".join(f"<th>{escape(str(col))}</th>" for col in columns)
    body_rows = []
    for row in rows:
        cells = "".join(
            f"<td>{escape(' '.join(str(row.get(col, '')).split()))}</td>"
            for col in columns
        )
        body_rows.append(f"<tr>{cells}</tr>")
    body_html = "".join(body_rows)
    st.markdown(
        f"""
        <div class="swiss-table-wrap" style="display:inline-block;">
            <table class="{table_class}">
                <thead><tr>{header_html}</tr></thead>
                <tbody>{body_html}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.divider()

try:
    now_metadata = load_metadata("nowcasting")
    fore_metadata = load_metadata("forecasting")
except FileNotFoundError as e:
    st.error(f"Could not find metadata: {e}")
    st.stop()

# ---- 1. Ridge Regression ----
st.markdown("### 1. For Regression Nowcasting: Ridge")
st.markdown("This is the best regression model (i.e. in predicting the percentage of delayed flights) under the nowcasting approach.  \n"
            "One of the aims of the nowcasting approach is to ascertain the dominant features, which requires the model to be easily interpretable.  \n"
            "Thus, while its accuracy (_R²_ = 0.5172) is slightly lower than the more complex models like Neural Network (_R²_ = 0.5427), this trade-off is justifiable due to its high interpretability.")

# Plot the flowchart
st.markdown("Figure 1 visualises the schematic workflow of the delay rate prediction using the Ridge model.  \n"
            "The parameters of this model are the weighting coefficients, _w_, and the biases, _b_; one pair of _w_ and _b_ for each feature.  \n"
            "The linearity of this model (no interaction between features), is the reason for its interpretability.")
image_path = os.path.join(PROJECT_ROOT, "app", "images", "diagram_linear.svg")
st.image(image_path, width=480)
st.caption("**Figure 1.** Schematic diagram of the prediction workflow using Ridge model.")
st.space('small')



## Prepare Model interpretation

_FEATURE_DESCRIPTIONS = {
    "delay_rate_lag1":           "Delay rate 1 month ago",
    "delay_rate_lag12":          "Delay rate 12 months ago",
    "rainy_days_arr_exp":        "Number of rainy days at arrival airport (exponential)",
    "temp_volatility_total_exp": "Temperature swing at departure and arrival airports (exponential)",
    "delay_rate_gradient":       "Month-on-month change in delay rate (trend)",
    "airline_Jetstar":           "Airline indicator: Jetstar",
    "airline_Qantas":            "Airline indicator: Qantas",
    "n_public_holidays_total":   "Number of public holidays",
    "route_Brisbane_Perth":      "Route indicator: Brisbane to Perth",
    "pct_school_holiday":        "Proportion of days that are school holidays",
    "route_Melbourne_Brisbane":  "Route indicator: Melbourne to Brisbane",
    "route_Sydney_Hobart":       "Route indicator: Sydney to Hobart",
    "airline_Rex Airlines":      "Airline indicator: Rex Airlines",
    "month_cos":                 "Cosine encoding of calendar month",
    "month_sin":                 "Sine encoding of calendar month",
    "route_Sydney_Brisbane":     "Route indicator: Sydney to Brisbane",
    "route_Perth_Adelaide":      "Route indicator: Perth to Adelaide",
}

# This is to draw the bar plot inside the table cells
def _svg_bar(value, max_abs, bar_width=320, zero_pct=0.25):
    """Return an SVG bar + numeric label for a coefficient cell.
    zero_pct controls where the zero-line sits (0.25 = 25% from left).
    """
    colour = COL["positive"] if value >= 0 else COL["negative"]
    zero_x = int(bar_width * zero_pct)
    pos_space = bar_width - zero_x   # pixels available for positive bars
    neg_space = zero_x               # pixels available for negative bars
    if max_abs > 0:
        fill_px = int(abs(value) / max_abs * (pos_space if value >= 0 else neg_space))
    else:
        fill_px = 0
    x = zero_x if value >= 0 else zero_x - fill_px
    label = f"{value:+.4f}"
    return (
        f'<div style="display:flex;align-items:center;gap:6px;white-space:nowrap;">'
        f'<svg width="{bar_width}" height="14" style="flex-shrink:0;">'
        f'<rect x="{x}" y="2" width="{fill_px}" height="10" fill="{colour}" rx="1"/>'
        f'<line x1="{zero_x}" y1="0" x2="{zero_x}" y2="14" stroke="rgba(13,13,13,0.2)" stroke-width="1"/>'
        f'</svg>'
        f'<span style="font-size:0.8rem;font-variant-numeric:tabular-nums;color:#0d0d0d;">{label}</span>'
        f'</div>'
    )

# This is to extract the coefficients from the saved model parameters
def _ridge_coef_section(models_dir, metadata, max_abs=None):
    try:
        ridge_path = os.path.join(models_dir, "ridge_regressor.pkl")
        ridge_model = load_joblib_model(ridge_path)
        coef = getattr(ridge_model, "coef_", None)
        feature_names = metadata.get("feature_names", [])
        if coef is not None and len(feature_names) == len(coef):
            coef_data = [
                {"Feature": name, "Coefficient": float(value), "Abs Coef": abs(float(value))}
                for name, value in zip(feature_names, coef)
            ]
            coef_data = sorted(coef_data, key=lambda x: x["Abs Coef"], reverse=True)[:10]
            coef_data = sorted(coef_data, key=lambda x: x["Coefficient"], reverse=True)
            if max_abs is None:
                max_abs = coef_data[0]["Abs Coef"] if coef_data else 1.0

            table_rows = []
            for rank, row in enumerate(coef_data, 1):
                table_rows.append({
                    "Rank": rank,
                    "Feature": row["Feature"],
                    "Description": _FEATURE_DESCRIPTIONS.get(row["Feature"], "—"),
                    "Weight Coefficient": _svg_bar(row["Coefficient"], max_abs),
                })

            # Render table with raw HTML in the Coefficient cell (skip escaping for that col)
            columns = ["Rank", "Feature", "Description", "Weight Coefficient"]
            header_html = "".join(f"<th>{escape(col)}</th>" for col in columns)
            body_rows = []
            for row in table_rows:
                cells = []
                for col in columns:
                    val = row.get(col, "")
                    if col == "Weight Coefficient":
                        cells.append(f"<td style='text-align:center;padding:0.46rem 0.62rem;'>{val}</td>")
                    else:
                        cells.append(f"<td>{escape(' '.join(str(val).split()))}</td>")
                body_rows.append(f"<tr>{''.join(cells)}</tr>")
            body_html = "".join(body_rows)
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
    except FileNotFoundError:
        st.warning(f"Ridge model file not found: `{models_dir}/ridge_regressor.pkl`")
    except Exception as exc:
        st.error("Failed to render Ridge coefficients.")
        st.code(str(exc), language="text")

#st.markdown("#### Model interpretation")
st.markdown("Since the input features are normalised, the relative contribution of each feature can be deduced by inspecting _w_, where:  \n"
            "- a larger _positive_ weight indicates that the feature contributes more to increase the delay rate;  \n"
            "- conversely, a larger _negative_ weight indicates the feature contributes more to _decrease_ the delay rate.  \n"
            )
st.markdown("The 10 most dominant features and their respective coefficients under the nowcasting approach are presented in Table 1 below.")

def _top10_max_abs(models_dir, metadata):
    try:
        ridge_model = load_joblib_model(os.path.join(models_dir, "ridge_regressor.pkl"))
        coef = getattr(ridge_model, "coef_", None)
        feature_names = metadata.get("feature_names", [])
        if coef is not None and len(feature_names) == len(coef):
            top10 = sorted(
                [abs(float(v)) for v in coef], reverse=True
            )[:10]
            return top10[0] if top10 else 1.0
    except Exception:
        pass
    return 1.0


# Plot the feature tables
#st.space('small')
st.caption("**Table.1** The top 10 features contributing to the current months delay rate and their respective weight coefficients.")
_ridge_coef_section(NOWCASTING_MODELS_DIR, now_metadata)
st.space('small')
# st.markdown("**FORECASTING**")
# _ridge_coef_section(FORECASTING_MODELS_DIR, fore_metadata)

st.markdown("It is observed that:  \n"
            "- For the strongest positive feature, `delay_rate_lag1`: when the delay rate is high in a particular month, it is likely to remain high in the next month as well.  \n"
            "   _This may suggest persisting operational problems, such as crew shortage or unplanned maintenance issues._  \n"
            "- For the strongest negative feature, `delay_rate_gradient`: when delay rate has been worsening month-to-month, it improves the likelihood that it will be lower the next month.  \n"
            "   _This may indicate that airlines notice the worsening trend, and then take appropriate measures to combat the problem._"
            )

###

# ---- 2. XGBoost Classification ----
st.divider()
st.markdown("### 2. For Classification Nowcasting and Forecasting: XGBoost")
st.markdown("This is the best classification model (i.e. in predicting if it is a high delay rate month) under both the nowcasting and forecasting approaches.  \n"
            "Its accuracy (_F1_ = 0.7525 for nowcasting, _F1_ = 0.7415 for forecasting) is slightly lower than the Neural Network model (_F1_ = 0.7618 for nowcasting, _F1_ = 0.7620 for forecasting).  \n"
            "However, after inspecting the accuracy breakdown, XGBoost exhibits the best balance between _precision_ (a measure of how well the model avoids false positives) and _recall_ (how well the model avoids false negatives)."
            )
st.markdown("For example, the precision and recall for the XGBoost model are both moderately high at 0.7475 and 0.7357, respectively, while the corresponding values for the Neural Network model are imbalanced at 0.6744 and 0.8757.  \n"
            "The lower precision score means the model is prone to classifying a month as high delay even when it is not the case, which is not desirable as it may lead to unnecessary additional spending in anticipation of the high delay."
            )
st.markdown("")

try:
    xgb_path = os.path.join(NOWCASTING_MODELS_DIR, "xgb_classifier.pkl")
    xgb_model = load_joblib_model(xgb_path)
    params = xgb_model.get_params()

    n_estimators = int(params.get("n_estimators", 0))
    max_depth = params.get("max_depth", "unknown")
    learning_rate = params.get("learning_rate", "unknown")
    min_child_weight = params.get("min_child_weight", "unknown")

    image_path = os.path.join(PROJECT_ROOT, "app", "images", "diagram_xgb.svg")
    st.image(image_path, width=500)
    st.caption("**Figure 2**. Schematic diagram of the prediction workflow using XGBoost model.")

    st.space('small')
    st.markdown("The hyperparameters of the XGBoost models, shown in Table 2 below, are obtained from the fine-tuning process using the validation dataset."
                )
    st.caption("**Table 2.** The hyperparameters of the XGBoost model  \n"
                )
    render_swiss_table(
        [
            {"Parameter": "n_estimators", "Value": n_estimators},
            {"Parameter": "max_depth", "Value": max_depth},
            {"Parameter": "learning_rate", "Value": learning_rate},
            {"Parameter": "min_child_weight", "Value": min_child_weight},
        ],
        columns=["Parameter", "Value"],
        table_class="swiss-table swiss-table-tight",
    )
except FileNotFoundError:
    st.warning(f"XGBoost model file not found: `{NOWCASTING_MODELS_DIR}/xgb_classifier.pkl`")
except Exception as exc:
    st.error("Failed to render XGBoost architecture.")
    st.code(str(exc), language="text")


###

# ---- 3. Neural Network Models ----
st.divider()
st.subheader("3. For Regression Forecasting: Neural Network Models")
st.markdown("This is the best regression model under the forecasting approach, because it has the highest accuracy (_R²_ = 0.5096) vs the other two models (_R²_ = 0.4931 and 0.5052).  \n"
    "Unlike the nowcasting approach (where interpretability is important), the accuracy of prediction is the main priority when forecasting for the upcoming month.")
st.markdown("The neural network architecture used to predict the delay rate under the nowcasting and forecasting approaches (identical in both) are presented in Table 3 as a list of the employed neural network layers."
            )
try:
    nn_reg_path = os.path.join(NOWCASTING_MODELS_DIR, "nn_regressor.keras")
    nn_clf_path = os.path.join(NOWCASTING_MODELS_DIR, "nn_classifier.keras")
    missing = [p for p in [nn_reg_path, nn_clf_path] if not os.path.exists(p)]

    if missing:
        st.warning(
            f"Neural Network artefacts not found at `{NOWCASTING_MODELS_DIR}`. "
            "Retrain the nowcasting model to generate NN artefacts."
        )
    else:
        nn_reg = load_keras_model(nn_reg_path)
        nn_clf = load_keras_model(nn_clf_path)

        col1, col2 = st.columns(2)
        with col1:
            st.caption("**Table 3.** The layers employed in the Neural Network model for both the nowcasting and forecasting approaches.")
            render_swiss_table(
                get_layer_rows(nn_reg),
                columns=["Layer", "Type", "Units", "Activation", "Dropout", "Params"],
            )

        # with col2:
        #     st.caption("**Table 4.** The layers employed in the classification Neural Network model.")
        #     render_swiss_table(
        #         get_layer_rows(nn_clf),
        #         columns=["Layer", "Type", "Units", "Activation", "Dropout", "Params"],
        #     )

except Exception as exc:
    st.error("Failed to render Neural Network architecture.")
    st.code(str(exc), language="text")
