"""
Model architecture dashboard.
Visualises architecture for Ridge, XGBoost, and Neural Network models.
"""

import json
import os
from html import escape

import joblib
import streamlit as st
import tensorflow as tf
from src.ui_theme import apply_theme


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NOWCASTING_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "nowcasting")
FORECASTING_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "forecasting")


st.set_page_config(page_title="FLAPS — Model Architecture", page_icon="✈️", layout="wide")
apply_theme(
    extra_css="""
    .swiss-note {
        max-width: 920px;
    }
    [data-testid="stMetric"] {
        min-height: 120px;
    }
    .arch-node {
        border: 1px solid rgba(13, 13, 13, 0.25);
        background: #f8f7f3;
        min-height: 74px;
        padding: 0.7rem 0.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        line-height: 1.3;
        font-size: 0.88rem;
    }
    .arch-arrow {
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 74px;
        font-size: 1.2rem;
        color: #0d0d0d;
    }
    .arch-arrow-vertical {
        text-align: center;
        font-size: 1.1rem;
        margin: 0.2rem 0;
        color: #0d0d0d;
    }
    .arch-stack {
        max-width: 520px;
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
        width: 100%;
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
    .swiss-table th:last-child,
    .swiss-table td:last-child {
        text-align: right;
        width: 180px;
        font-variant-numeric: tabular-nums;
    }
    """
)
st.title("Model Architecture")
st.markdown(
    "This page shows the architectures of the top 3 best performing machine learning models. "
    "Please select between the Nowcasting and Forecasting models using the selection below."
)


def get_models_dir(model_set):
    return NOWCASTING_MODELS_DIR if model_set == "nowcasting" else FORECASTING_MODELS_DIR


@st.cache_data
def load_metadata(model_set):
    path = os.path.join(get_models_dir(model_set), "metadata.json")
    with open(path, "r") as f:
        return json.load(f)


@st.cache_resource
def load_joblib_model(path):
    return joblib.load(path)


@st.cache_resource
def load_keras_model(path):
    return tf.keras.models.load_model(path, compile=False)


def render_flow_row(nodes):
    n_nodes = len(nodes)
    cols = st.columns((n_nodes * 2) - 1)
    for idx, node in enumerate(nodes):
        with cols[idx * 2]:
            st.markdown(f'<div class="arch-node">{node}</div>', unsafe_allow_html=True)
        if idx < n_nodes - 1:
            with cols[(idx * 2) + 1]:
                st.markdown('<div class="arch-arrow">-></div>', unsafe_allow_html=True)


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


def render_nn_stack(model):
    input_dim = model.input_shape[-1] if getattr(model, "input_shape", None) else "?"
    st.markdown(
        f'<div class="arch-node">Input<br>{input_dim} features</div>',
        unsafe_allow_html=True,
    )

    for idx, row in enumerate(get_layer_rows(model)):
        st.markdown('<div class="arch-arrow-vertical">|</div>', unsafe_allow_html=True)
        label = f"{row['Layer']}<br>{row['Type']}"
        if row["Units"] != "-":
            label += f" ({row['Units']})"
        if row["Activation"] != "-":
            label += f"<br>activation={row['Activation']}"
        if row["Dropout"] != "-":
            label += f"<br>dropout={row['Dropout']}"
        st.markdown(f'<div class="arch-node">{label}</div>', unsafe_allow_html=True)
        if idx == len(model.layers) - 1:
            st.markdown('<div class="arch-arrow-vertical">|</div>', unsafe_allow_html=True)


def render_swiss_table(rows, columns):
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
        <div class="swiss-table-wrap">
            <table class="swiss-table">
                <thead><tr>{header_html}</tr></thead>
                <tbody>{body_html}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


model_set_label = st.radio(
    "Model Set",
    options=["Nowcasting", "Forecasting"],
    horizontal=True,
    help="Choose which trained artefacts to inspect.",
)
model_set = model_set_label.lower()
models_dir = get_models_dir(model_set)

try:
    metadata = load_metadata(model_set)
except FileNotFoundError:
    st.error(f"Could not find metadata at `{models_dir}/metadata.json`.")
    st.stop()

st.caption(
    f"Loaded: `{model_set}` artefacts | "
    f"Features: {len(metadata.get('feature_names', []))}"
)


st.subheader("1. Ridge Regression")
st.markdown('<div class="swiss-section-kicker">Pipeline</div>', unsafe_allow_html=True)
render_flow_row([
    f"Input features<br>{len(metadata.get('feature_names', []))} columns",
    "StandardScaler<br>Z-score normalisation",
    "Ridge linear model<br>y_hat = w^T x + b",
    "Predicted delay rate",
])
st.markdown('<div class="pipeline-gap"></div>', unsafe_allow_html=True)

try:
    ridge_path = os.path.join(models_dir, "ridge_regressor.pkl")
    ridge_model = load_joblib_model(ridge_path)
    alpha = getattr(ridge_model, "alpha", "unknown")
    n_features = getattr(ridge_model, "n_features_in_", len(metadata.get("feature_names", [])))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Alpha (L2 strength)", f"{alpha}")
    with col2:
        st.metric("Input Features", f"{n_features}")
    with col3:
        st.metric("Fit Intercept", str(getattr(ridge_model, "fit_intercept", True)))

    coef = getattr(ridge_model, "coef_", None)
    feature_names = metadata.get("feature_names", [])
    if coef is not None and len(feature_names) == len(coef):
        coef_rows = [
            {"Feature": name, "Coefficient": float(value), "Abs Coef": abs(float(value))}
            for name, value in zip(feature_names, coef)
        ]
        coef_rows = sorted(coef_rows, key=lambda x: x["Abs Coef"], reverse=True)[:15]
        for row in coef_rows:
            row["Coefficient"] = f"{row['Coefficient']:+.4f}"
            row.pop("Abs Coef", None)
        st.markdown("**Top Coefficients (|w|)**")
        render_swiss_table(coef_rows, columns=["Feature", "Coefficient"])
except FileNotFoundError:
    st.warning(f"Ridge model file not found: `{models_dir}/ridge_regressor.pkl`")
except Exception as exc:
    st.error("Failed to render Ridge architecture.")
    st.code(str(exc), language="text")


st.divider()
st.subheader("2. XGBoost Classification")
#st.caption("Lightweight architecture view (no per-tree inspection).")

try:
    xgb_path = os.path.join(models_dir, "xgb_classifier.pkl")
    xgb_model = load_joblib_model(xgb_path)
    params = xgb_model.get_params()

    n_estimators = int(params.get("n_estimators", 0))
    max_depth = params.get("max_depth", "unknown")
    learning_rate = params.get("learning_rate", "unknown")
    min_child_weight = params.get("min_child_weight", "unknown")

    st.markdown('<div class="swiss-section-kicker">Ensemble Flow</div>', unsafe_allow_html=True)
    render_flow_row([
        "Input features",
        "Tree 1",
        "Tree 2",
        "...",
        f"Tree {n_estimators}",
        "Additive score",
        "Sigmoid",
        "P(high delay)",
    ])
    st.markdown('<div class="pipeline-gap"></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("n_estimators", f"{n_estimators}")
    with c2:
        st.metric("max_depth", f"{max_depth}")
    with c3:
        st.metric("learning_rate", f"{learning_rate}")
    with c4:
        st.metric("min_child_weight", f"{min_child_weight}")
except FileNotFoundError:
    st.warning(f"XGBoost model file not found: `{models_dir}/xgb_classifier.pkl`")
except Exception as exc:
    st.error("Failed to render XGBoost architecture.")
    st.code(str(exc), language="text")


st.divider()
st.subheader("3. Neural Network Models")
#st.caption("Layer stack diagrams and architecture tables (no Graphviz).")

try:
    nn_reg_path = os.path.join(models_dir, "nn_regressor.keras")
    nn_clf_path = os.path.join(models_dir, "nn_classifier.keras")
    missing = [p for p in [nn_reg_path, nn_clf_path] if not os.path.exists(p)]

    if missing:
        st.warning(
            f"Neural network artefacts not found for `{model_set}` at `{models_dir}`. "
            "Retrain that model set to generate NN artefacts."
        )
    else:
        nn_reg = load_keras_model(nn_reg_path)
        nn_clf = load_keras_model(nn_clf_path)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Regression Network**")
            render_nn_stack(nn_reg)
            st.markdown("**Layer Table (Regression)**")
            render_swiss_table(
                get_layer_rows(nn_reg),
                columns=["Layer", "Type", "Units", "Activation", "Dropout", "Params"],
            )

        with col2:
            st.markdown("**Classification Network**")
            render_nn_stack(nn_clf)
            st.markdown("**Layer Table (Classification)**")
            render_swiss_table(
                get_layer_rows(nn_clf),
                columns=["Layer", "Type", "Units", "Activation", "Dropout", "Params"],
            )

except Exception as exc:
    st.error("Failed to render Neural Network architecture.")
    st.code(str(exc), language="text")
