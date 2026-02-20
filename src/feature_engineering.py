"""
Feature engineering pipeline for FLAPS multi-route model.

Extracts and transforms features from the merged flight+weather+holiday dataset
into the 38-feature matrix used by the ML models. This replicates the logic
from notebook 8b_model_multiroute_1.1.ipynb.
"""

import numpy as np
import pandas as pd


# Routes excluded due to anomalous test-set behaviour (see notebooks 6c, 6e)
ANOMALOUS_ROUTES = ['Melbourne_Hobart', 'Adelaide_Sydney', 'Perth_Brisbane']

# Minimum average flights/month for an airline-route to be included (see notebook 6d)
VOLUME_THRESHOLD = 50


def add_derived_columns(df):
    """Add airline_route, route, year, month_num, year_month_dt columns if missing."""
    df = df.copy()
    if 'year_month_dt' not in df.columns:
        df['year_month_dt'] = pd.to_datetime(df['year_month'])
    if 'month_num' not in df.columns:
        df['month_num'] = df['year_month_dt'].dt.month
    if 'year' not in df.columns or df['year'].dtype == float:
        df['year'] = df['year_month_dt'].dt.year
    df['year'] = df['year'].astype(int)
    if 'route' not in df.columns:
        df['route'] = df['departing_port'] + '_' + df['arriving_port']
    if 'airline_route' not in df.columns:
        df['airline_route'] = df['airline'] + '_' + df['departing_port'] + '_' + df['arriving_port']
    df = df.sort_values(['airline_route', 'year_month_dt']).reset_index(drop=True)
    return df


def filter_low_volume(df, threshold=VOLUME_THRESHOLD):
    """Exclude airline-routes with average flights/month below threshold."""
    airline_route_volume = df.groupby('airline_route')['sectors_scheduled'].mean()
    high_volume = airline_route_volume[airline_route_volume >= threshold].index.tolist()
    return df[df['airline_route'].isin(high_volume)].copy()


def filter_anomalous_routes(df, routes=None):
    """Exclude routes with anomalous test-set behaviour."""
    if routes is None:
        routes = ANOMALOUS_ROUTES
    return df[~df['route'].isin(routes)].copy()


def compute_lag_features(df):
    """Compute lagged delay rate features per airline-route."""
    df = df.copy()
    df['delay_rate_lag1'] = df.groupby('airline_route')['delay_rate'].shift(1)
    df['delay_rate_lag2'] = df.groupby('airline_route')['delay_rate'].shift(2)
    df['delay_rate_gradient'] = df['delay_rate_lag1'] - df['delay_rate_lag2']
    return df


def compute_weather_lag_features(df):
    """
    Lag weather features by 1 month for forecasting mode.

    For true forecasting, same-month weather is unavailable. This function
    creates lagged versions of weather features (previous month's weather).
    """
    df = df.copy()
    weather_cols = ['rainy_days_arr', 'temp_volatility_total', 'extreme_weather_days_total']
    for col in weather_cols:
        if col in df.columns:
            df[f'{col}_lag1'] = df.groupby('airline_route')[col].shift(1)
    return df


def compute_weather_climatology(df, train_years=None):
    """
    Compute historical monthly averages for weather features (climatology).

    For forecasting, use long-term seasonal averages instead of same-month
    observations. Computes averages from training years only to prevent leakage.

    Parameters
    ----------
    df : DataFrame
        Must contain route, month_num, and weather columns.
    train_years : list or None
        Years to use for computing climatology. If None, uses default
        training years (2010-2017 + 2023).

    Returns
    -------
    df : DataFrame with climatology columns added
    """
    df = df.copy()

    if train_years is None:
        train_years = list(range(2010, 2018)) + [2023]

    weather_cols = ['rainy_days_arr', 'temp_volatility_total', 'extreme_weather_days_total']

    # Compute climatology from training data only
    train_mask = df['year'].isin(train_years)

    for col in weather_cols:
        if col in df.columns:
            # Group by route and month to get seasonal average per route
            climatology = df[train_mask].groupby(['route', 'month_num'])[col].mean()
            # Map back to full dataset
            df[f'{col}_climatology'] = df.set_index(['route', 'month_num']).index.map(
                lambda x: climatology.get(x, np.nan)
            )

    return df


def compute_weather_transforms(df, max_values=None):
    """
    Compute transformed weather features.

    Parameters
    ----------
    df : DataFrame
        Must contain rainy_days_arr, temp_volatility_dep, temp_volatility_arr,
        extreme_weather_days_dep, extreme_weather_days_arr.
    max_values : dict or None
        If provided, use these max values for normalisation (for inference).
        If None, compute from the data (for training) and return them.

    Returns
    -------
    df : DataFrame with new columns added
    max_values : dict of max values used for normalisation
    """
    df = df.copy()

    # Compute raw totals first
    df['temp_volatility_total'] = df['temp_volatility_dep'] + df['temp_volatility_arr']
    df['extreme_weather_days_total'] = df['extreme_weather_days_dep'] + df['extreme_weather_days_arr']

    if max_values is None:
        max_values = {
            'rainy_days_arr_max': df['rainy_days_arr'].max(),
            'temp_volatility_total_max': df['temp_volatility_total'].max(),
        }

    # Guard against zero/NaN max values to avoid divide-by-zero and overflow.
    rainy_max = max_values.get('rainy_days_arr_max')
    temp_vol_max = max_values.get('temp_volatility_total_max')
    if not np.isfinite(rainy_max) or rainy_max <= 0:
        rainy_max = 1.0
    if not np.isfinite(temp_vol_max) or temp_vol_max <= 0:
        temp_vol_max = 1.0
    max_values['rainy_days_arr_max'] = rainy_max
    max_values['temp_volatility_total_max'] = temp_vol_max

    df['rainy_days_arr_exp'] = np.exp(df['rainy_days_arr'] / rainy_max)
    df['temp_volatility_total_exp'] = np.exp(df['temp_volatility_total'] / temp_vol_max)

    return df, max_values


def compute_cyclical_month(df):
    """Add sin/cos cyclical encoding for month."""
    df = df.copy()
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
    return df


def one_hot_encode(df):
    """
    One-hot encode airline and route columns.

    Returns
    -------
    df : DataFrame with dummy columns appended
    airline_cols : list of airline dummy column names
    route_cols : list of route dummy column names
    """
    df = df.copy()
    airline_dummies = pd.get_dummies(df['airline'], prefix='airline')
    route_dummies = pd.get_dummies(df['route'], prefix='route')
    df = pd.concat([df, airline_dummies, route_dummies], axis=1)
    return df, list(airline_dummies.columns), list(route_dummies.columns)


def build_feature_matrix(df, max_values=None, expected_airlines=None, expected_routes=None):
    """
    Full feature engineering pipeline: filter, lag, transform, encode.

    Parameters
    ----------
    df : DataFrame
        Raw merged training data (from ml_training_data_multiroute_hols.csv).
    max_values : dict or None
        Max values for weather exp transforms. None = compute from data.
    expected_airlines : list or None
        If provided, ensure these airline dummy columns exist (for inference).
    expected_routes : list or None
        If provided, ensure these route dummy columns exist (for inference).

    Returns
    -------
    dict with keys:
        'df' : processed DataFrame (with all columns)
        'feature_names' : list of 38 feature column names
        'airline_cols' : list of airline dummy column names
        'route_cols' : list of route dummy column names
        'max_values' : dict of max values used for exp transforms
    """
    # Add derived columns
    df = add_derived_columns(df)

    # Filter
    df = filter_low_volume(df)
    df = filter_anomalous_routes(df)

    # Lag features
    df = compute_lag_features(df)

    # Lag12 (same month last year)
    df = df.copy()
    df['delay_rate_lag12'] = df.groupby('airline_route')['delay_rate'].shift(12)

    # Weather transforms
    df, max_values = compute_weather_transforms(df, max_values=max_values)

    # Cyclical month
    df = compute_cyclical_month(df)

    # One-hot encoding
    df, airline_cols, route_cols = one_hot_encode(df)

    # Ensure expected columns exist (for inference with unseen airlines/routes)
    if expected_airlines is not None:
        for col in expected_airlines:
            if col not in df.columns:
                df[col] = 0
        airline_cols = expected_airlines
    if expected_routes is not None:
        for col in expected_routes:
            if col not in df.columns:
                df[col] = 0
        route_cols = expected_routes

    # Drop rows with NaN lag features (including lag12)
    df = df.dropna(subset=[
        'delay_rate_lag1', 'delay_rate_lag2', 'delay_rate_gradient',
        'delay_rate_lag12',
    ]).copy()

    # Assemble feature list (order matters — must match training)
    feature_names = (
        airline_cols
        + route_cols
        + ['month_sin', 'month_cos', 'delay_rate_lag1', 'sectors_scheduled']
        + ['rainy_days_arr_exp', 'delay_rate_lag12', 'delay_rate_gradient',
           'temp_volatility_total_exp', 'extreme_weather_days_total']
        + ['n_public_holidays_total', 'pct_school_holiday']
    )

    return {
        'df': df,
        'feature_names': feature_names,
        'airline_cols': airline_cols,
        'route_cols': route_cols,
        'max_values': max_values,
    }

## Training data for forecasting ##
# Need a new training matrix
def build_forecasting_feature_matrix(df, max_values=None, expected_airlines=None,
                                     expected_routes=None):
    """
    Feature engineering pipeline for the forecasting model (notebook 15a).

    Key differences from build_feature_matrix():
    - Adds delay_rate_lag12 (same month last year)
    - Excludes same-month weather features (unavailable at prediction time)
    - Drops rows missing lag12

    Returns dict with same keys as build_feature_matrix().
    """
    # Add derived columns
    df = add_derived_columns(df)

    # Filter
    df = filter_low_volume(df)
    df = filter_anomalous_routes(df)

    # Lag features (lag1, lag2, gradient)
    df = compute_lag_features(df)

    # Lag12 (same month last year)
    df = df.copy()
    df['delay_rate_lag12'] = df.groupby('airline_route')['delay_rate'].shift(12)

    # Weather transforms (needed so columns exist in df, but not in feature list)
    df, max_values = compute_weather_transforms(df, max_values=max_values)

    # Cyclical month
    df = compute_cyclical_month(df)

    # One-hot encoding
    df, airline_cols, route_cols = one_hot_encode(df)

    # Ensure expected columns exist (for inference with unseen airlines/routes)
    if expected_airlines is not None:
        for col in expected_airlines:
            if col not in df.columns:
                df[col] = 0
        airline_cols = expected_airlines
    if expected_routes is not None:
        for col in expected_routes:
            if col not in df.columns:
                df[col] = 0
        route_cols = expected_routes

    # Drop rows with NaN lag features (including lag12)
    df = df.dropna(subset=[
        'delay_rate_lag1', 'delay_rate_lag2', 'delay_rate_gradient',
        'delay_rate_lag12',
    ]).copy()

    # Assemble feature list — no weather features for forecasting
    feature_names = (
        airline_cols
        + route_cols
        + ['month_sin', 'month_cos', 'delay_rate_lag1', 'sectors_scheduled']
        + ['delay_rate_lag12', 'delay_rate_gradient']
        + ['n_public_holidays_total', 'pct_school_holiday']
    )

    return {
        'df': df,
        'feature_names': feature_names,
        'airline_cols': airline_cols,
        'route_cols': route_cols,
        'max_values': max_values,
    }


def split_data(df, feature_names):
    """
    Split into train/val/test using year-based stratification.

    Train: 2010-2017 + 2023
    Val:   2018 + 2024
    Test:  2019 + 2025+

    Returns dict with X_train, X_val, X_test, y_train_reg, y_val_reg, y_test_reg,
    y_train_clf, y_val_clf, y_test_clf, and the boolean masks.
    """
    train_mask = (((df['year'] >= 2010) & (df['year'] <= 2017)) | (df['year'] == 2023))
    val_mask = ((df['year'] == 2018) | (df['year'] == 2024))
    test_mask = ((df['year'] == 2019) | (df['year'] >= 2025))

    X_train = df.loc[train_mask, feature_names].values
    X_val = df.loc[val_mask, feature_names].values
    X_test = df.loc[test_mask, feature_names].values

    y_train_reg = df.loc[train_mask, 'delay_rate'].values
    y_val_reg = df.loc[val_mask, 'delay_rate'].values
    y_test_reg = df.loc[test_mask, 'delay_rate'].values

    y_train_clf = df.loc[train_mask, 'is_high_delay'].values
    y_val_clf = df.loc[val_mask, 'is_high_delay'].values
    y_test_clf = df.loc[test_mask, 'is_high_delay'].values

    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train_reg': y_train_reg, 'y_val_reg': y_val_reg, 'y_test_reg': y_test_reg,
        'y_train_clf': y_train_clf, 'y_val_clf': y_val_clf, 'y_test_clf': y_test_clf,
        'train_mask': train_mask, 'val_mask': val_mask, 'test_mask': test_mask,
    }
