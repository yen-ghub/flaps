# Project FLAPS — Flight Lateness Australia Prediction System

A machine learning system for predicting monthly flight delay rates on Australian domestic routes.

**The full project is available as an interactive web application at [flaps.online](https://flaps.online).**

---

## Overview

Australian domestic on-time performance fell from ~81% (2010–2019) to 63–68% in 2023–2024. FLAPS is a proof-of-concept ML system that predicts monthly delay rates for airline-route combinations using publicly available flight performance data (BITRE) and weather observations ([BOM](http://www.bom.gov.au/)), trained on 15+ years of data (2010–2026, excluding COVID).

---

## Explore the App

Visit **[flaps.online](https://flaps.online)** and navigate through the sidebar:

| Page | What you will find |
|------|-----------------|
| **Nowcasting** | Run nowcast prediction of past month's flight delay, to assess model performance and identify the dominant predictive features |
| **Forecasting** | Run forecast prediction of this month's flight delay, available from the first day of the month |
| **Model Details** | Architecture, diagrams, and feature importance for the top 3 performing ML models |
| **Model Performance** | Metrics, baseline comparisons, time-series plots, and confusion matrices |
| **Update and Re-train** | Download the latest data and retrain models live |

---

## Methods & Techniques

- **Data acquisition** — automated pipelines across three sources: FTP (BOM), HTTP (BITRE), and REST API (Flightera)
- **Feature engineering** — hand-crafted weather, holiday, lag, and seasonality features; hypothesis-tested one at a time
- **Hypothesis-driven experimentation** — each feature evaluated individually, results documented step-by-step in Jupyter notebooks
- **Time-series validation** — train/val/test stratification across pre- and post-COVID periods to prevent temporal leakage
- **Model evaluation** — ML models benchmarked (Ridge, Random Forest, Logistic Regression, XGBoost, Neural Network); best model selected per task based on interpretability vs. accuracy trade-off
- **Problem diagnosis** — identified and resolved model degradation caused by low-volume routes (< 50 flights/month) whose delay rates are statistically too noisy to predict reliably
- **Trade-off consideration** — simpler models preferred when added complexity does not justify the accuracy gains
- **End-to-end deployment** — raw data to web-deployed interactive application on Google Cloud Run

---

## Tech Stack

Python · scikit-learn · XGBoost · TensorFlow/Keras · pandas · Streamlit · Google Cloud Run · Docker

---

## Repository Structure

```
src/            # Data acquisition, feature engineering, model training
app/            # Streamlit web application (5 pages)
notebooks/      # Exploratory analysis and experiment logs
models/         # Pre-trained model artefacts (nowcasting + forecasting)
data/           # Raw and processed datasets
```
