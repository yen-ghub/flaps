"""
Shared Streamlit UI theme helpers.
"""

import streamlit as st


_BASE_THEME_CSS = """
[data-testid="stDivider"] {
    border-top: 1px solid rgba(13, 13, 13, 0.25);
}
.swiss-note {
    max-width: 900px;
    color: #5a5a5a;
    font-size: 1rem;
    line-height: 1.5;
    margin-bottom: 1.2rem;
}
.swiss-section-kicker {
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #5a5a5a;
    margin-bottom: 0.3rem;
}
[data-testid="stMetric"] {
    background: transparent;
    border: 1px solid rgba(13, 13, 13, 0.25);
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
    border: 1px solid rgba(13, 13, 13, 0.25);
}
[data-testid="stSidebar"] {
    background: #efeee9;
    width: 210px !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    background: #efeee9;
    border-right: 1px solid rgba(13, 13, 13, 0.25);
    width: 210px !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
    border-radius: 0 !important;
    color: #0d0d0d;
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
    border-left: 2px solid #0d0d0d;
    padding-left: calc(0.45rem - 2px);
}
[data-testid="stSidebar"] hr {
    border-color: rgba(13, 13, 13, 0.25);
    margin: 0.9rem 0;
}
[data-testid="stSidebar"] .stSelectbox label {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.72rem;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    border-radius: 0;
    border: 1px solid rgba(13, 13, 13, 0.25);
    background: #f8f7f3;
    min-height: 42px;
}
"""


def apply_theme(extra_css: str = "") -> None:
    """Inject shared CSS with optional page-specific additions."""
    css = _BASE_THEME_CSS
    if extra_css:
        css = f"{css}\n{extra_css.strip()}\n"
    st.markdown(f"<style>\n{css}</style>", unsafe_allow_html=True)
