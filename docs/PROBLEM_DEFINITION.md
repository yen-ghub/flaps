# Problem Definition: Australian Domestic Flight Delay Prediction System

## Table of Contents

1. [Executive summary](#1-executive-summary)
2. [Problem statement](#2-problem-statement)
3. [Data sources & constraints](#3-data-sources-and-constraints)
4. [Modelling approach](#4-modelling-approach)
5. [Success criteria](#5-success-criteria)
6. [Assumptions and constraints](#6-assumptions-and-constraints)
7. [Use cases](#7-use-cases)
8. [Future improvements](#8-future-improvements)
9. [Glossary](#9-glossary)
10. [References](#10-references)

## 1. Executive summary

This project develops a machine learning system to predict monthly flight delay rates for Australian domestic aviation. Using publicly available data from the Bureau of Infrastructure and Transport Research Economics (BITRE), Bureau of Meteorology (BOM), and Airservices Australia, the system predicts the percentage of flights that will be delayed on specific route-airline combinations.

The system employs a **hybrid modeling approach**: 
1. **Primary:** Regression model predicting exact monthly delay rates (0-100%)
2. **Secondary:** Binary classification model identifying high-delay months

This approach demonstrates working with real-world data constraints, thoughtful problem framing, and production-oriented thinking.

---

## 2. Problem statement

### 2.1 Background

According to BITRE data, Australian domestic flight delays are on a higher than average level. During the 2023-2024 period, only 63-68% of domestic flights arrive on-time (within 15 minutes of schedule), compared to the long-term average of 81%. Cancellation rate is at 5%, which is also higher than the historical average of 2.1%. These delays incur significant cost to the airlines and negatively impact passenger experience.

Delay prediction based on weather statistics, however, is far from straightforward as the relevant information is only available after the period of the interest. Both flight performance data and monthly aggregate weather data become available only after the month ends. While this creates challenges for real-time prediction, it provides an opportunity to build and validate models using complete historical data before developing a true forecasting system.

### 2.2 Problem Definition
**The Challenge:**

Airlines, passengers, and airport operations currently rely on historical delay statistics published monthly after the fact. There is no established system to predict future delay patterns before they occur.

**What needs prediction:**

Monthly delay rate (percentage of flights delayed >15 minutes) for specific route-airline combinations. The 15-minute threshold follows the definition set by BITRE.

**The constraint:**

Both flight performance data and weather data are published only after the month concludes. We cannot use February's actual weather to predict February's delays in real-time.

### 2.3 Objectives

This project has one primary and one secondary objectives:

**Primary Objective (Regression):**
> "Predict the monthly delay rate (percentage of flights delayed >15 minutes) for a given route-airline combination?"

**Secondary Problem (Classification):**  
> "Predict which route-airline-month combination experience high delays (delay rate >25%)?"

### 2.4 Scope

**In Scope:**
- Australian domestic commercial flights
- Major carriers: Qantas, Virgin Australia, Jetstar, QantasLink
- Monthly predictions at route-airline granularity
- Data from 2010
- Arrival delays

**Out of Scope:**
- Individual flight predictions (data not available)
- Real-time/daily predictions
- International flights
- Cancellation prediction
- Causal analysis of delay sources

---

## 3. Data sources and constraints

### 3.1 Available Data

**BITRE Flight Performance Data**
- **URL:** [BITRE Time Series](https://www.bitre.gov.au/sites/default/files/documents/OTP_Time_Series_Master_Current_september_2025.xlsx)
- **Format:** Excel *.xlsx*, monthly aggregates by route and airline
- **Date range:** November 2003 - present
- **Variables:** Total flights, on-time arrivals, delays, cancellations

**Example data structure:**
```
Month: January 2024
Route: Sydney-Melbourne  
Airline: Qantas
Total Flights: 930
On-Time: 744
Delayed: 186
→ Delay Rate: 20.0%
```

**BOM Weather Data:**
- **Format:** CSV, daily at airport weather station, to be aggregated manually
- **Source:** http://www.bom.gov.au/climate/data/
- **Date range:** 2009 - present
- **Variables:** Daily rainfall totals, max temperatures, average wind speed etc.


### 3.2 Data constraints

**Temporal resolution limitation:** Since the flight delay data are monthly aggregates, predicting the delay for specific flight at a specific time is not possible. We predict the overall monthly delay rate for a route.

**Forecasting challenge:** Both BITRE flight data and BOM weather data are published monthly after the month concludes. For example:
- January 2024 flight performance: available February 1st
- January 2024 weather data: available February 1st

This creates a fundamental constraint: **current month's weather data is not available to predict current month's delays in real-time**, i.e. straight up forecasting of the flight delay rate is not possible.

---

## 4. Modelling approach

As a solution to the data constraints outlined in Section 3.2, a two-part approach is adopted.

1. Part 1: Nowcasting (Learning Phase)
    - Use complete monthly data (weather + delays from same month)
    - Build and validate model architecture
    - Can achieve higher accuracy, but does not predict future events
    - Understand which features have the most significant effects  
    - Example: Use January 2024 weather to explain January 2024 delays (both available February 1st).

2. Part 2: Forecasting (Production Phase)
    - Predict next month using only historical data
    - Use lagged features: previous months' delays and weather
    - Use seasonal patterns: typical weather for target month
    - Lower accuracy but genuinely predictive
    - Example: On January 31st, predict February 2024 delays using data through January only.

Recall that there are two objectives within each Part (see Section 2.3), where each objective requires a different algorithm: 
* **Primary: Regression Model**   
  Predicting continuous delay rates preserves maximum information. An output of "23.4%" is more actionable than "high" or "low" for operational planning.

  **Algorithms to evaluate:**
  - Linear Regression (baseline)    
  - XGBoost Regressor
  - LightGBM Regressor
  - Random Forest Regressor


  **Target variable:**
  ```python
  delay_rate = (Total_Flights - On_Time_Arrivals) / Total_Flights
  ```

* **Secondary: Classification Model**
  Binary prediction of high-delay months can be derived from regression outputs or trained separately.

  **Algorithms to evaluate:**
  - XGBoost Classifier
  - Random Forest Classifier
  - Logistic Regression (baseline)

  **Target variable:**
  ```python
  is_high_delay = (delay_rate > 0.25).astype(int)  # 25% threshold
  ```

### 4.3 Validation Strategy

Time-based cross-validation is critical to prevent temporal leakage:

```
Training        :   Jan 2020 - Jun 2023
Cross-validation:   Jul 2023 - Dec 2023
Test            :   Jan 2024 - Jun 2024
```

Stratification by airline and season ensures representative splits.

---

## 5. Success Criteria

### 5.1 Part 1: Nowcasting Model

| Metric | Good | Excellent |
|--------|------|-----------|
| MAE | < 0.05 | < 0.03 |
| RMSE | < 0.08 | < 0.05 |
| R² | > 0.40 | > 0.55 |

**Interpretation:** MAE < 0.05 means average error less than 5 percentage points. If true rate is 20%, predicting 15-25% is acceptable.

**Baseline to beat:** Historical average (MAE ~0.08-0.10)

### 5.2 Part 2: Forecasting Model (1-Month Ahead)

| Metric | Good | Excellent |
|--------|------|-----------|
| MAE | < 0.06 | < 0.04 |
| RMSE | < 0.10 | < 0.07 |
| R² | > 0.30 | > 0.45 |

**Why lower targets:** Predicting next month without current weather is harder. Lower R² is expected but still demonstrates predictive skill above baseline.

**Baseline to beat:** Previous month's rate (MAE ~0.06-0.08)

### 5.3 Classification Model (Both Parts)

| Metric | Good | Excellent |
|--------|------|-----------|
| Accuracy | > 0.70 | > 0.80 |
| Precision | > 0.65 | > 0.75 |
| Recall | > 0.60 | > 0.70 |
| F1 | > 0.65 | > 0.75 |
| AUC-ROC | > 0.75 | > 0.85 |

**Baseline to beat:** Majority class prediction (~60-70% accuracy)

---

## 6. Assumptions and Constraints

### 6.1 Key Assumptions

1. **Historical patterns are predictive:** Past delay patterns contain signal about future delays
2. **Monthly aggregates sufficient:** Route-level monthly patterns are stable enough to predict
3. **Weather impact:** Monthly weather aggregates correlate with flight delays
4. **Data quality:** BITRE data accurately reflects on-time performance (15-minute threshold)
5. **Temporal stability:** Airline operational patterns remain relatively stable year-to-year (excluding COVID period)
6. **Excludable factors:** Some delay causes (mechanical issues, crew problems) are not directly observable but reflected in historical patterns

### 6.2 Constraints

**Data Constraints:**
- Monthly aggregates only (no individual flights)
- Limited historical weather variables (only basic metrics available monthly)
- COVID-19 data gap (April-December 2020)
- Route coverage limited to major routes with sufficient traffic

**Technical Constraints:**
- Computational: Personal laptop / free cloud tier (limited GPU)
- Time: 10-12 weeks for complete project
- Budget: $0 (or minimal cloud costs if needed)

**Scope Constraints:**
- Portfolio project, not production deployment
- Domestic Australian flights only
- Focusing on post-2020 data for consistency

### 6.3 Known Limitations

**What We Cannot Capture:**
- Real-time operational issues (mechanical failures, crew delays)
- Air traffic control congestion (not in available data)
- Cascading delay effects within a single day
- Airline-specific operational decisions
- Security incidents or other one-off events

**Impact on Model:**
- R² unlikely to exceed 0.60 due to unobservable factors
- Model best at identifying systematic patterns, not random events
- Predictions represent expected delay rate, not guarantee

---

## 7. Use cases

**Historical Analysis (Part 1):**
- Understand what weather/temporal factors drive delays
- Validate model before deploying for forecasting
- Benchmark airline performance

**Operational Planning (Part 2):**
- Airlines allocate resources 1 month ahead
- Passengers book more reliable routes
- Airports prepare for high-delay periods

---

## 8. Future Improvements

**Enhanced Modeling:**
- Multi-class classification (Low/Medium/High/Severe delay categories)
- Separate model for cancellation prediction
- Ensemble methods combining multiple models
- Deep learning approaches (LSTM for time series)

**Additional Features:**
- Aircraft type distributions by route
- Fuel price trends (airline cost pressure indicator)
- Airport construction/renovation schedules
- Special event calendars (sports, concerts, conferences)

**Product Enhancements:**
- Real-time prediction API
- Interactive dashboard with Plotly/Streamlit
- Mobile-friendly web interface
- Integration with flight booking sites

**Advanced Analytics:**
- Explainability dashboard (SHAP values visualization)
- Confidence intervals for predictions
- Sensitivity analysis (what-if scenarios)
- Comparative analytics (route benchmarking)

**Production Readiness:**
- Automated retraining pipeline
- Model monitoring and drift detection
- A/B testing framework
- Scalable deployment (Docker/Kubernetes)

---

## 9. Glossary

- **On-Time Performance:** Flight arrives/departs within 15 minutes of scheduled time
- **Delay Rate:** Percentage of flights delayed >15 minutes  
- **Route:** City pair (e.g., Sydney-Melbourne)
- **Monthly Aggregate:** Summary statistics for an entire month
- **Temporal Leakage:** Using future information to predict the past (fatal error)
- **MAE:** Mean Absolute Error (average prediction error)
- **BITRE:** Bureau of Infrastructure and Transport Research Economics
- **BOM:** Bureau of Meteorology

---

## 10. References

**Data Sources:**
- BITRE On-Time Performance: https://www.bitre.gov.au/statistics/aviation/otphome
- BOM Climate Data: http://www.bom.gov.au/climate/data/
- OurAirports: https://ourairports.com/data/

**Technical Resources:**
- Scikit-learn Documentation: https://scikit-learn.org/
- XGBoost Documentation: https://xgboost.readthedocs.io/
- Time Series Validation: "The Correct Way to Do Cross-Validation in Time Series"

**Domain Knowledge:**
- BITRE Aviation Statistics Yearbook
- Australian airline industry reports (ACCC)
- Flight delay prediction academic papers (to be added)

### Contact

For questions about this project:
- Email: yendrew.y@gmail.com

---

*Last Updated: 14 January 2026*