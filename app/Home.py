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
##
st.subheader('Why')
st.markdown("""
            The on-time performance of Australian domestic flights used to be at 81% (over the 2010-2019 period).  
            Surprisingly, it declined to 63-68% in 2023-2024.
            
            What is going on?  

            On the premise that better predictions could lead to better planning for airlines and airports, the idea for Project FLAPS (Flight Lateness Australia Prediction System) was born. 
            """)
st.space('small')

st.subheader('What')
st.markdown("""            
            Project FLAPS is designed to be a proof-of-concept model to test the feasibility of using machine learning models in predicting the delay rates on Australian domestic flight routes.  
            It uses historical data on flight performance and weather observations for training purposes.  
            Historical data for daily weather observations published by the Bureau of Meteorology ([BOM](http://www.bom.gov.au/)) are free and publicly accessible.

            However, historical data for daily flight performance need to be purchased and they are expensive (e.g. from Airservices Australia or Flightradar24). 
            To overcome this issue of high costs and data availability, the free _monthly_ flight performance data published by the Bureau of Infrastructure and Transport Research Economics ([BITRE](https://www.bitre.gov.au/)) are used.  
            The implication is that only _monthly_ delay rates can be predicted, not _daily_ delay rates.

            Nonetheless, if decent predictions can be made for monthly delay rates, it is assumed that accurate predictions can also be made for daily delay rates (with some modifications) if access to historical data for daily flight performance is available (e.g. through the usage of proprietary or purchased data).
            """)
st.space('small')

##
st.subheader('How')
st.markdown("""  
            Based on the above, Project FLAPS set out to build machine learning models to predict the monthly delay rates on Australian domestic flight routes, as a proof-of-concept.
            """)
# st.space('small')
# st.markdown("  \n")
st.markdown("##### General")
st.markdown("""
            Firstly, 7 machine learning models were selected based on their level of complexity, consisting of:
            - 3 regression models – Ridge, Random Forest and Neural Network – used to predict the monthly delay rate [(i.e. the percentage of delayed flights in the month)].
            - 4 classification models – Logistic, Random Forest, XGBoost and Neural Network – used to predict if the next month will be a high delay month (defined as >25% of delayed flights in the month).

            The simpler and more interpretable models were used as a starting point, before progressing with more complex models, which may yield better accuracy but were less interpretable.  
            All 7 machine learning models were trained (including validation and testing) on a dataset spanning more than 15 years of data (2010-2026, excluding the COVID period) from the BOM and BITRE.

            Secondly, feature selection and engineering were performed using a hypothesis testing approach.  
            Potential features with high correlation with the target value (i.e. the monthly delay rate) were first selected.  
            Each potential feature was then tested individually to ascertain its impact to the model performance.  
            It was retained if there was measurable improvement, otherwise it was omitted.
            """)
#Each potential feature was tested in isolation and retained only when a measurable improvement was observed.
# st.space('xxsmall')
st.markdown("  \n")
st.markdown("##### Data Complications")
st.markdown("""
            Towards the latter part of the development Project FLAPS, it was discovered that the monthly flight performance data published by BITRE lacked timeliness for the purposes of this project.  
            The data for a certain month is typically released by BITRE towards the end of the following month (e.g. the monthly flight performance data for January 2026 was only released on 20 February 2026).  
            Such timing made it difficult to make a meaningful forecast for the month ahead.

            Thus, the following two-part approach is developed:  
            1. **Nowcasting approach** 
                - The purpose of this approach is to aid with model selection and identification of dominant features.  
                - No forecast will be made for the current month.
                - This approach uses real-time data (i.e. applying the model to a completed month using that month's own data, rather than predicting ahead) and thus is only applicable to previous months where data is available.
                - Accordingly, each of the 7 machine learning models will utilise data up until the previous month to provide insight on the previous month's flight delays.
                - For example: each model uses data up until January 2026 to _provide insight on_ January 2026's delays itself.
            2. **Forecasting approach**
                - The purpose of this approach is to use data up until the previous month to forecast the flight delay rate for the current month.
                - A forecast is made for the current month.
                - However, due to the aforementioned delayed publication of data by BITRE, such forecast will typically only be available towards the end of the current month.  
                    For example: each model uses January 2026 data to predict February 2026 delays, if using data published by BITRE, the forecast would only be available from 20 February 2026.  
                - To address the lack of timeliness in using the data published by BITRE, an additional source of monthly flight performance data is obtained from the [Flightera Flight Data API](https://rapidapi.com/flightera/api/flightera-flight-data), which is available immediately after the month has ended.  
                    For example: each model uses January 2026 data to predict February 2026 delays, if using data published by Flightera Flight Data API, the forecast would be available from 1 February 2026.
            
            Integrating the most recent month’s flight performance data from Flightera Flight Data API addresses the timing constraint presented by the BITRE data.  
            While it is a paid source, fortunately the cost for obtaining such data is relatively low and competitively priced.  
            To validate, the delay rates obtained from Flightera and BITRE were compared to make sure they are consistent.  
            This was confirmed by comparing their delay rates for January 2026, where the reported values for the Sydney to Melbourne route (as a representative sample) from these two sources only differ by less than 2 %.
            """)
# st.space('xxsmall')
st.markdown("  \n")
st.markdown("""
            ##### Best Overall Models

            Depending on the approach (nowcasting or forecasting) and the type of prediction (regression or classification), there are three best overall models identified.  
            Each of these models are briefly described below:
            - For **regression nowcasting**: Ridge.  
                It is a simple linear model, but the performance is still comparable to the more complex non-linear models.  
                Although there is a small sacrifice in accuracy, it is worth the trade-off in higher interpretability -- which is a priority in the nowcasting approach.
            - For both **classification nowcasting** and **forecasting**: XGBoost.  
                It gives the best balance between precision and recall amongst the other classification models.  
                As the regression model provided interpretability, the focus here is on achieving a good, balanced performance.  
                _Good precision_ means when the model predicts a high-delay month, it is likely to be actually a high-delay month.  
                _Good recall_ means the model does not miss many actual high-delay months.  
                When relying solely on good precision, there is a potential risk the model _predicts a high-delay month only when it is very certain_ and misses out on many actual high-delay months.  
                On the other hand, when relying solely on good recall, there is a potential risk that the model _predicts a high-delay month even when it is not certain_ in order to avoid missing out.  
                In general, a reliable model needs to have a good balance of both.
            - For **regression forecasting**: Neural Network. 
                It gives the best accuracy out of all the regression models under the forecasting approach.  
                The model is more complex and less interpretable, but this trade-off is acceptable because the priority of the forecasting approach is the prediction accuracy.
            
            These models are discussed in further detail on the [Model Details](Model_Details) page and evaluated on the [Model Evaluation](Model_Evaluation) page.
            """)

st.markdown("  \n")
st.markdown("""
            ##### Tech Stack

            Python · pandas · scikit-learn · XGBoost · TensorFlow/Keras · Jupyter · Streamlit · Docker · Google Cloud Run
            """
            )

st.divider()


# --- Skills Demonstrated section ---
st.subheader("Skills Demonstrated")

def _skill(header, description):
    return (
        f"- **{header}**<br>"
        f'<span style="font-style:italic; color:#5a5a5a;">{description}</span>'
    )

_skills_left = [
    ("Data acquisition",                  "Development of automated pipelines to collect and combine data from multiple acquisition methods and sources: FTP for BOM data, webpage link for BITRE data and REST API for Flightera data."),
    ("Feature engineering",               "Hand-engineered and encoded features, including weather, holidays, past delay rates, and seasonality."),
    ("Hypothesis-driven experimentation", "Systematic feature evaluation, which is documented step-by-step in Jupyter notebooks."),
    ("Time-series validation",            "Training, validation and testing data stratification between pre- and post-Covid periods, to prevent temporal leakage."),
]

_skills_right = [
    ("Model evaluation",      "Performance comparison of 7 machine learning models and validation against baseline benchmarks."),
    ("Problem diagnosis",     "Identification of root cause when the model performance deteriorates, e.g. filtering out low-volume flights because their data is inherently too noisy for prediction."),
    ("Trade-off consideration","Preference for simpler models when the added complexity of more sophisticated models does not justify the gains."),
    ("End-to-end deployment", "From raw data to interactive web application deployed on Google Cloud Run."),
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


