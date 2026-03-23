# Project FLAPS — Flight Lateness Australia Prediction System

A machine learning system for predicting monthly flight delay rates on Australian domestic routes using publicly available data.

**The full project is available as an interactive web application at [flaps.online](https://flaps.online).**


## Why

The on-time performance of Australian domestic flights used to be at 81% (over the 2010-2019 period). Surprisingly, it declined to 63-68% in 2023-2024.

What is going on?

On the premise that better predictions could lead to better planning for airlines and airports, the idea for Project FLAPS (Flight Lateness Australia Prediction System) was born.


## What

Project FLAPS is designed to be a proof-of-concept model to test the feasibility of using machine learning models in predicting the delay rates on Australian domestic flight routes. It uses historical data on flight performance and weather observations for training purposes. Historical data for daily weather observations published by the Bureau of Meteorology ([BOM](http://www.bom.gov.au/)) are free and publicly accessible.

However, historical data for daily flight performance need to be purchased and they are expensive (e.g. from Airservices Australia or Flightradar24). To overcome this issue of high costs and data availability, the free _monthly_ flight performance data published by the Bureau of Infrastructure and Transport Research Economics ([BITRE](https://www.bitre.gov.au/)) are used. The implication is that only _monthly_ delay rates can be predicted, not _daily_ delay rates.

Nonetheless, if decent predictions can be made for monthly delay rates, it is assumed that accurate predictions can also be made for daily delay rates (with some modifications) if access to historical data for daily flight performance is available (e.g. through the usage of proprietary or purchased data).

## How

Based on the above, Project FLAPS set out to build machine learning models to predict the monthly delay rates on Australian domestic flight routes, as a proof-of-concept.

### General

Firstly, 7 machine learning models were selected based on their level of complexity, consisting of:
- 3 regression models – Ridge, Random Forest and Neural Network – used to predict the monthly delay rate [(i.e. the percentage of delayed flights in the month)].
- 4 classification models – Logistic, Random Forest, XGBoost and Neural Network – used to predict if the next month will be a high delay month (defined as >25% of delayed flights in the month).

The simpler and more interpretable models were used as a starting point, before progressing with more complex models, which may yield better accuracy but were less interpretable. All 7 machine learning models were trained (including validation and testing) on a dataset spanning more than 15 years of data (2010-2026, excluding the COVID period) from the BOM and BITRE.

Secondly, feature selection and engineering were performed using a hypothesis testing approach. Potential features with high correlation with the target value (i.e. the monthly delay rate) were first selected. Each potential feature was then tested individually to ascertain its impact to the model performance. It was retained if there was measurable improvement, otherwise it was omitted.

### Data Complications

Towards the latter part of the development Project FLAPS, it was discovered that the monthly flight performance data published by BITRE lacked timeliness for the purposes of this project. The data for a certain month is typically released by BITRE towards the end of the following month (e.g. the monthly flight performance data for January 2026 was only released on 20 February 2026). Such timing made it difficult to make a meaningful forecast for the month ahead.

Thus, the following two-part approach is developed:  
1. **Nowcasting approach** 
    - The purpose of this approach is to aid with model selection and identification of dominant features.  
    - No forecast will be made for the current month.
    - This approach uses real-time data (i.e. [explain general meaning or description of technique briefly]) and is only applied to previous months where data is available.
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

#### Best Overall Models

Depending on the approach (nowcasting or forecasting) and the type of prediction (regression or classification), there are three best overall models identified.  
Each of these models are briefly described below and discussed in further detail on the [Model Details](https://flaps.online/Model_Details) page of the web app:
- For **regression nowcasting**: Ridge.
    It is a simple linear model, but the performance is still comparable to the more complex non-linear models. Although there is a small sacrifice in accuracy, it is worth the trade-off in higher interpretability; which is a priority in the nowcasting approach. The highest-accuracy route is Sydney→Melbourne (R² = 0.703).
- For both **classification nowcasting** and **forecasting**: XGBoost.
    It gives the best balance between precision and recall amongst the other classification models. As the regression model provided interpretability, the focus here is on achieving a good, balanced performance. On the test set, the model achieves precision = 0.754 & recall = 0.751 for the nowcasting approach, and precision = 0.756 =& recall = 0.735 for the forecasting approach.

    _Good precision_ means when the model predicts a high-delay month, it is likely to be actually a high-delay month.
    _Good recall_ means the model does not miss many actual high-delay months.

    When relying solely on good precision, there is a potential risk the model _predicts a high-delay month only when it is very certain_ and misses out on many actual high-delay months.
    On the other hand, when relying solely on good recall, there is a potential risk that the model _predicts a high-delay month even when it is not certain_ in order to avoid missing out.
    In general, a reliable model needs to have a good balance of both.
- For **regression forecasting**: Neural Network.
    It gives the best accuracy out of all the regression models under the forecasting approach. The highest-accuracy route is Adelaide→Perth (R² = 0.652).
    The model is more complex and less interpretable, but this trade-off is acceptable because the priority of the forecasting approach is the prediction accuracy.


## Tech Stack

Python · pandas · scikit-learn · XGBoost · TensorFlow/Keras · Jupyter · Streamlit · Docker · Google Cloud Run


## Skills Demonstrated

- **Data acquisition** — automated pipelines across three sources: FTP (BOM), HTTP (BITRE), and REST API (Flightera)
- **Feature engineering** — hand-crafted weather, holiday, lag, and seasonality features; hypothesis-tested one at a time
- **Hypothesis-driven experimentation** — each feature evaluated individually, results documented step-by-step in Jupyter notebooks
- **Time-series validation** — train/val/test stratification across pre- and post-COVID periods to prevent temporal leakage
- **Model evaluation** — ML models benchmarked (Ridge, Random Forest, Logistic Regression, XGBoost, Neural Network); best model selected per task based on interpretability vs. accuracy trade-off
- **Problem diagnosis** — identified and resolved model degradation caused by low-volume routes (< 50 flights/month) whose delay rates are statistically too noisy to predict reliably
- **Trade-off consideration** — simpler models preferred when added complexity does not justify the accuracy gains
- **End-to-end deployment** — raw data to interactive web application deployed on Google Cloud Run


## Explore the App

Visit **[flaps.online](https://flaps.online)** and navigate through the sidebar:

| Page | What you will find |
|------|-----------------|
| **Nowcasting** | Run nowcast prediction of past month's flight delay, to assess model performance and identify the dominant predictive features |
| **Forecasting** | Run forecast prediction of this month's flight delay, available from the first day of the month |
| **Model Details** | Architecture, diagrams, and feature importance for the top 3 performing ML models |
| **Model Performance** | Metrics, baseline comparisons, time-series plots, and confusion matrices |
| **Update and Re-train** | Download the latest data and retrain models live |


## Repository Structure

```
├── app/                      # Streamlit UI for predictions
├── data/                     # Raw and processed data
├── docs/                     # Problem definition and documentation
├── models/                   # Saved model artefacts
├── notebooks/                # Step-by-step development notebooks
├── src/                      # Training and preprocessing scripts
├── requirements.txt
└── README.md
```
