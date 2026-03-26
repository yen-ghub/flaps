"""
Unit tests for src/feature_engineering.py

Covers: compute_lag_features, filter_low_volume, split_data, build_feature_matrix
Run with: pytest tests/test_feature_engineering.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add the project root to the path so we can import from src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_engineering import (
    VOLUME_THRESHOLD,
    compute_lag_features,
    filter_low_volume,
    split_data,
    build_feature_matrix,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TRAINING_CSV = os.path.join(PROJECT_ROOT, 'data', 'processed', 'ml_training_data_multiroute_hols.csv')


# ---------------------------------------------------------------------------
# compute_lag_features
# ---------------------------------------------------------------------------

def test_compute_lag_features_correct_values():
    """Lag values equal the previous row's delay_rate within each airline-route group."""
    # Two airline-routes, each with 3 months of data in chronological order.
    # After lagging, the first row of each group should have no prior value (NaN), and subsequent rows should pick up the previous month's delay rate.
    df = pd.DataFrame({
        'airline_route': ['A_X_Y', 'A_X_Y', 'A_X_Y', 'B_X_Y', 'B_X_Y', 'B_X_Y'],
        'delay_rate':    [0.10,    0.20,    0.30,    0.40,    0.50,    0.60],
    })
    result = compute_lag_features(df)

    # Group A: first row has no history, rows 1 and 2 look back one step
    assert pd.isna(result.loc[0, 'delay_rate_lag1'])
    assert result.loc[1, 'delay_rate_lag1'] == pytest.approx(0.10)
    assert result.loc[2, 'delay_rate_lag1'] == pytest.approx(0.20)

    # Group B: same pattern — the shift resets at the group boundary
    assert pd.isna(result.loc[3, 'delay_rate_lag1'])
    assert result.loc[4, 'delay_rate_lag1'] == pytest.approx(0.40)
    assert result.loc[5, 'delay_rate_lag1'] == pytest.approx(0.50)


def test_compute_lag_features_no_bleed_across_groups():
    """lag1 of the first row of group B must not use the last row of group A."""
    # This is the key correctness property of a grouped shift, the lag should never carry over from one airline-route into another.
    # If groupby is applied correctly, the first row of group B is always NaN regardless of what group A looks like.
    df = pd.DataFrame({
        'airline_route': ['A_X_Y', 'A_X_Y', 'B_X_Y', 'B_X_Y'],
        'delay_rate':    [0.10,    0.20,    0.30,    0.40],
    })
    result = compute_lag_features(df)

    # If groupby is missing, index 2 would incorrectly get lag1=0.20 (from group A).
    # It should be NaN — the start of a new airline-route has no prior month.
    assert pd.isna(result.loc[2, 'delay_rate_lag1'])


def test_compute_lag_features_gradient():
    """delay_rate_gradient equals lag1 minus lag2."""
    # The gradient is a momentum feature: it captures whether delays are trending up or down.
    # It requires both lag1 and lag2, so only the third row onward will have a non-NaN value.
    df = pd.DataFrame({
        'airline_route': ['A_X_Y', 'A_X_Y', 'A_X_Y'],
        'delay_rate':    [0.10,    0.30,    0.50],
    })
    result = compute_lag_features(df)

    # Row 2: lag1=0.30, lag2=0.10, so gradient = 0.30 - 0.10 = 0.20
    assert result.loc[2, 'delay_rate_gradient'] == pytest.approx(0.20)
    # Rows 0 and 1 don't have enough history — gradient should be NaN
    assert pd.isna(result.loc[0, 'delay_rate_gradient'])
    assert pd.isna(result.loc[1, 'delay_rate_gradient'])


# ---------------------------------------------------------------------------
# filter_low_volume
# ---------------------------------------------------------------------------

def test_filter_low_volume_keeps_above_threshold():
    """Routes at or above VOLUME_THRESHOLD are kept; routes below are dropped."""
    # Three airline-routes with different average flight volumes:
    #   'high'     → avg 60, well above threshold → keep
    #   'boundary' → avg 50, exactly at threshold → keep (filter is >=)
    #   'low'      → avg 49, just below threshold → drop
    df = pd.DataFrame({
        'airline_route':      ['high'] * 3 + ['boundary'] * 3 + ['low'] * 3,
        'sectors_scheduled':  [60, 60, 60,   50, 50, 50,   49, 49, 49],
    })
    result = filter_low_volume(df)

    assert set(result['airline_route'].unique()) == {'high', 'boundary'}
    assert 'low' not in result['airline_route'].values


def test_filter_low_volume_empty_when_all_below_threshold():
    """Returns an empty DataFrame (not an error) when all routes are below threshold."""
    # Edge case: if no airline-route clears the volume bar, the result should be an empty DataFrame rather than raising an exception.
    df = pd.DataFrame({
        'airline_route':     ['A'] * 3 + ['B'] * 3,
        'sectors_scheduled': [10, 10, 10, 20, 20, 20],
    })
    result = filter_low_volume(df)

    assert len(result) == 0


# ---------------------------------------------------------------------------
# split_data
# ---------------------------------------------------------------------------

def _make_split_df(years, months=None):
    """Minimal DataFrame with the columns split_data requires."""
    n = len(years)
    return pd.DataFrame({
        'year':          years,
        # Default to June so the Jan 2026 special case doesn't interfere
        'month':         months if months is not None else [6] * n,
        'delay_rate':    [0.2] * n,
        'is_high_delay': [0] * n,
        'feature':       [1.0] * n,
    })


def test_split_data_year_assignment():
    """Years are assigned to the correct train / val / test splits."""
    # The split design reflects a pre/post-COVID stratification strategy:
    #   Train: 2010-2017 (pre-COVID baseline) + 2023 (post-COVID recovery)
    #   Val:   2018 + 2024
    #   Test:  2019 + 2025
    # COVID years (2020-2022) are excluded from all splits.
    years = [2010, 2014, 2018, 2019, 2023, 2024, 2025]
    df = _make_split_df(years)
    splits = split_data(df, feature_names=['feature'])

    train_years = set(df.loc[splits['train_mask'], 'year'])
    val_years   = set(df.loc[splits['val_mask'],   'year'])
    test_years  = set(df.loc[splits['test_mask'],  'year'])

    assert {2010, 2014, 2023} <= train_years
    assert {2018, 2024} == val_years
    assert {2019, 2025} <= test_years


def test_split_data_jan_2026_is_test():
    """January 2026 goes to the test split; other months of 2026 do not."""
    # The test set includes one extra month beyond 2025: January 2026.
    # This captures the most recent available data point. February 2026 onward
    # is not assigned to any split (it would be future data for inference).
    df = _make_split_df(years=[2026, 2026], months=[1, 2])
    splits = split_data(df, feature_names=['feature'])

    assert splits['test_mask'].iloc[0] == True   # Jan 2026 → test
    assert splits['test_mask'].iloc[1] == False  # Feb 2026 → nowhere


def test_split_data_covid_years_excluded():
    """COVID years (2020-2022) are not assigned to any split."""
    # The COVID period is excluded because anomalous travel restrictions make those years unrepresentative of normal operating conditions.
    # Rows from 2020-2022 should simply fall through all three masks.
    df = _make_split_df(years=[2020, 2021, 2022])
    splits = split_data(df, feature_names=['feature'])

    assert splits['X_train'].shape[0] == 0
    assert splits['X_val'].shape[0] == 0
    assert splits['X_test'].shape[0] == 0


# ---------------------------------------------------------------------------
# build_feature_matrix (integration smoke test)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(TRAINING_CSV), reason="Training CSV not available")
def test_build_feature_matrix_shape_and_no_nan():
    """Feature matrix columns match feature_names list and contain no NaN values."""
    # This is an end-to-end smoke test on the real training data.
    # It checks that the full pipeline (filtering, lag computation, weather transforms, encoding) produces a clean matrix that matches the declared feature list.
    # Any mismatch in feature count or unexpected NaN would break downstream model training.
    df = pd.read_csv(TRAINING_CSV)
    result = build_feature_matrix(df)

    feature_df = result['df'][result['feature_names']]

    # The number of columns in the matrix must match the feature_names list exactly
    assert feature_df.shape[1] == len(result['feature_names'])

    # No NaN should remain — lag rows with insufficient history are dropped by the pipeline
    assert not feature_df.isna().any().any(), "NaN values found in feature matrix"

    # Sanity-check that the two most important lag features made it in
    assert 'delay_rate_lag1' in result['feature_names']
    assert 'delay_rate_lag12' in result['feature_names']
