"""
Train all FLAPS models and save artifacts to models/.

Replicates notebook 8b_model_multiroute_1.1.ipynb:
- Loads ml_training_data_multiroute_hols.csv
- Runs feature engineering pipeline
- Trains Ridge, RF Reg, Logistic, RF Clf, XGBoost
- Trains two sets of models: nowcasting and forecasting
- Saves models, scaler, and metadata

Usage:
    python -m src.train_and_save
"""

import json
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    HAS_TF = False

from src.data_loader import FORECASTING_MODELS_DIR, NOWCASTING_MODELS_DIR, load_training_data, load_load_factor_data
from src.feature_engineering import build_feature_matrix, build_forecasting_feature_matrix, split_data

## Nowcasting models ##
def train_and_save():
    """Run the full training pipeline for nowcasting and save all artifacts."""

    # --- Load data ---
    # The functions used here are from data_loader.py
    print("Loading training data...")
    df = load_training_data()
    print(f"  Shape: {df.shape}")

    # --- Feature engineering ---
    # The functions used here are from feature_engineering.py
    print("\nRunning feature engineering pipeline...")
    result = build_feature_matrix(df)
    df_processed = result['df']
    feature_names = result['feature_names']
    airline_cols = result['airline_cols']
    route_cols = result['route_cols']
    max_values = result['max_values']

    print(f"  Processed rows: {len(df_processed)}")
    print(f"  Features: {len(feature_names)}")

    # --- Train/val/test split ---
    print("\nSplitting data...")
    splits = split_data(df_processed, feature_names)
    print(f"  Train: {splits['X_train'].shape[0]}")
    print(f"  Val:   {splits['X_val'].shape[0]}")
    print(f"  Test:  {splits['X_test'].shape[0]}")

    X_train = splits['X_train']
    X_val = splits['X_val']
    X_test = splits['X_test']
    y_train_reg = splits['y_train_reg']
    y_val_reg = splits['y_val_reg']
    y_test_reg = splits['y_test_reg']
    y_train_clf = splits['y_train_clf']
    y_val_clf = splits['y_val_clf']
    y_test_clf = splits['y_test_clf']

    # --- Scale ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # --- Train models ---
    metrics = {}

    # Ridge Regression (scaled)
    print("\nTraining Ridge Regression...")
    ridge = Ridge(alpha=0.1)
    ridge.fit(X_train_scaled, y_train_reg)
    ridge_pred = ridge.predict(X_test_scaled)
    ridge_r2 = r2_score(y_test_reg, ridge_pred)
    ridge_rmse = np.sqrt(mean_squared_error(y_test_reg, ridge_pred))
    ridge_mae = mean_absolute_error(y_test_reg, ridge_pred)
    print(f"  Test R²: {ridge_r2:.4f}, RMSE: {ridge_rmse:.4f}, MAE: {ridge_mae:.4f}")
    metrics['ridge'] = {'R2': ridge_r2, 'RMSE': ridge_rmse, 'MAE': ridge_mae}

    # Random Forest Regression (unscaled)
    print("Training Random Forest Regression...")
    rf_reg = RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    rf_reg.fit(X_train, y_train_reg)
    rf_pred = rf_reg.predict(X_test)
    rf_r2 = r2_score(y_test_reg, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test_reg, rf_pred))
    rf_mae = mean_absolute_error(y_test_reg, rf_pred)
    print(f"  Test R²: {rf_r2:.4f}, RMSE: {rf_rmse:.4f}, MAE: {rf_mae:.4f}")
    metrics['rf_reg'] = {'R2': rf_r2, 'RMSE': rf_rmse, 'MAE': rf_mae}

    # Logistic Regression (scaled)
    print("Training Logistic Regression...")
    logreg = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    logreg.fit(X_train_scaled, y_train_clf)
    logreg_pred = logreg.predict(X_test_scaled)
    logreg_proba = logreg.predict_proba(X_test_scaled)[:, 1]
    logreg_f1 = f1_score(y_test_clf, logreg_pred)
    logreg_auc = roc_auc_score(y_test_clf, logreg_proba)
    logreg_precision = precision_score(y_test_clf, logreg_pred)
    logreg_recall = recall_score(y_test_clf, logreg_pred)
    print(f"  Test F1: {logreg_f1:.4f}, AUC: {logreg_auc:.4f}")
    metrics['logreg'] = {
        'F1': logreg_f1, 'AUC': logreg_auc,
        'Precision': logreg_precision, 'Recall': logreg_recall,
    }

    # Random Forest Classification (unscaled)
    print("Training Random Forest Classification...")
    rf_clf = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    rf_clf.fit(X_train, y_train_clf)
    rf_clf_pred = rf_clf.predict(X_test)
    rf_clf_proba = rf_clf.predict_proba(X_test)[:, 1]
    rf_clf_f1 = f1_score(y_test_clf, rf_clf_pred)
    rf_clf_auc = roc_auc_score(y_test_clf, rf_clf_proba)
    rf_clf_precision = precision_score(y_test_clf, rf_clf_pred)
    rf_clf_recall = recall_score(y_test_clf, rf_clf_pred)
    print(f"  Test F1: {rf_clf_f1:.4f}, AUC: {rf_clf_auc:.4f}")
    metrics['rf_clf'] = {
        'F1': rf_clf_f1, 'AUC': rf_clf_auc,
        'Precision': rf_clf_precision, 'Recall': rf_clf_recall,
    }

    # XGBoost Classification (unscaled)
    if HAS_XGB:
        print("Training XGBoost Classification...")
        xgb_clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            min_child_weight=5, random_state=42, n_jobs=-1,
        )
        xgb_clf.fit(X_train, y_train_clf, eval_set=[(X_val, y_val_clf)], verbose=False)
        xgb_pred = xgb_clf.predict(X_test)
        xgb_proba = xgb_clf.predict_proba(X_test)[:, 1]
        xgb_f1 = f1_score(y_test_clf, xgb_pred)
        xgb_auc = roc_auc_score(y_test_clf, xgb_proba)
        xgb_precision = precision_score(y_test_clf, xgb_pred)
        xgb_recall = recall_score(y_test_clf, xgb_pred)
        print(f"  Test F1: {xgb_f1:.4f}, AUC: {xgb_auc:.4f}")
        metrics['xgb_clf'] = {
            'F1': xgb_f1, 'AUC': xgb_auc,
            'Precision': xgb_precision, 'Recall': xgb_recall,
        }
    else:
        xgb_clf = None
        print("XGBoost not available, skipping.")

    # Neural Network (TensorFlow/Keras)
    nn_reg = None
    nn_clf = None
    nn_reg_pred = None
    nn_clf_proba = None
    nn_clf_pred = None

    if HAS_TF:
        tf.random.set_seed(42)
        np.random.seed(42)

        # Custom callback to print epoch progress
        class EpochProgressCallback(keras.callbacks.Callback):
            def __init__(self, model_name, total_epochs):
                super().__init__()
                self.model_name = model_name
                self.total_epochs = total_epochs

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                val_loss = logs.get('val_loss', 0)
                if (epoch+1)%10 == 0:
                    print(f"  {self.model_name} Epoch {epoch + 1}/{self.total_epochs} - val_loss: {val_loss:.4f}")

        # Neural Network Regression
        print("Training Neural Network Regression...")
        nn_reg = keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='linear')
        ])
        nn_reg.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                       loss='mse', metrics=['mae'])
        early_stop_reg = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True, verbose=0
        )
        epoch_progress_reg = EpochProgressCallback("NN Regression", 500)
        nn_reg.fit(
            X_train_scaled, y_train_reg,
            validation_data=(X_val_scaled, y_val_reg),
            epochs=500, batch_size=64,
            callbacks=[early_stop_reg, epoch_progress_reg],
            verbose=0
        )
        nn_reg_pred = nn_reg.predict(X_test_scaled, verbose=0).flatten()
        nn_reg_r2 = r2_score(y_test_reg, nn_reg_pred)
        nn_reg_rmse = np.sqrt(mean_squared_error(y_test_reg, nn_reg_pred))
        nn_reg_mae = mean_absolute_error(y_test_reg, nn_reg_pred)
        print(f"  Test R²: {nn_reg_r2:.4f}, RMSE: {nn_reg_rmse:.4f}, MAE: {nn_reg_mae:.4f}")
        metrics['nn_reg'] = {'R2': nn_reg_r2, 'RMSE': nn_reg_rmse, 'MAE': nn_reg_mae}

        # Neural Network Classification
        tf.random.set_seed(42)
        np.random.seed(42)
        print("Training Neural Network Classification...")
        nn_clf = keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        nn_clf.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                       loss='binary_crossentropy', metrics=['accuracy'])
        early_stop_clf = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True, verbose=0
        )
        epoch_progress_clf = EpochProgressCallback("NN Classification", 500)
        n_neg_now = int((y_train_clf == 0).sum())
        n_pos_now = int((y_train_clf == 1).sum())
        class_weight_now = {0: 1.0, 1: n_neg_now / max(n_pos_now, 1)}
        print(f"  Class weights: {class_weight_now}")
        nn_clf.fit(
            X_train_scaled, y_train_clf.astype(np.float32),
            validation_data=(X_val_scaled, y_val_clf.astype(np.float32)),
            epochs=500, batch_size=64,
            callbacks=[early_stop_clf, epoch_progress_clf],
            class_weight=class_weight_now,
            verbose=0
        )
        nn_clf_proba = nn_clf.predict(X_test_scaled, verbose=0).flatten()
        nn_clf_pred = (nn_clf_proba >= 0.5).astype(int)
        nn_clf_f1 = f1_score(y_test_clf, nn_clf_pred)
        nn_clf_auc = roc_auc_score(y_test_clf, nn_clf_proba)
        nn_clf_precision = precision_score(y_test_clf, nn_clf_pred)
        nn_clf_recall = recall_score(y_test_clf, nn_clf_pred)
        print(f"  Test F1: {nn_clf_f1:.4f}, AUC: {nn_clf_auc:.4f}")
        metrics['nn_clf'] = {
            'F1': nn_clf_f1, 'AUC': nn_clf_auc,
            'Precision': nn_clf_precision, 'Recall': nn_clf_recall,
        }
    else:
        print("TensorFlow not available, skipping neural network training.")

    # --- Compute route-level performance ---
    df_test = df_processed[splits['test_mask']].copy()
    df_test['ridge_pred'] = ridge.predict(X_test_scaled)
    df_test['rf_pred'] = rf_reg.predict(X_test)
    df_test['naive_lag1'] = df_test['delay_rate_lag1']
    if nn_reg_pred is not None:
        df_test['nn_pred'] = nn_reg_pred

    route_metrics = {}
    for route in sorted(df_test['route'].unique()):
        rd = df_test[df_test['route'] == route]
        route_metrics[route] = {
            'ridge_r2': float(r2_score(rd['delay_rate'], rd['ridge_pred'])),
            'rf_r2': float(r2_score(rd['delay_rate'], rd['rf_pred'])),
            'lag1_r2': float(r2_score(rd['delay_rate'], rd['naive_lag1'])),
            'n_samples': int(len(rd)),
        }
        if nn_reg_pred is not None:
            route_metrics[route]['nn_r2'] = float(r2_score(rd['delay_rate'], rd['nn_pred']))

    # Naive baselines (overall)
    overall_lag1_r2 = float(r2_score(y_test_reg, df_test['naive_lag1'].values))
    train_mean = float(y_train_reg.mean())

    # Feature importance from RF
    importance = {
        name: float(imp)
        for name, imp in zip(feature_names, rf_reg.feature_importances_)
    }

    # --- Save artifacts ---
    os.makedirs(NOWCASTING_MODELS_DIR, exist_ok=True)

    print("\nSaving models...")
    joblib.dump(ridge, os.path.join(NOWCASTING_MODELS_DIR, 'ridge_regressor.pkl'))
    joblib.dump(rf_reg, os.path.join(NOWCASTING_MODELS_DIR, 'rf_regressor.pkl'))
    joblib.dump(logreg, os.path.join(NOWCASTING_MODELS_DIR, 'logreg_classifier.pkl'))
    joblib.dump(rf_clf, os.path.join(NOWCASTING_MODELS_DIR, 'rf_classifier.pkl'))
    if xgb_clf is not None:
        joblib.dump(xgb_clf, os.path.join(NOWCASTING_MODELS_DIR, 'xgb_classifier.pkl'))
    joblib.dump(scaler, os.path.join(NOWCASTING_MODELS_DIR, 'scaler.pkl'))
    if nn_reg is not None:
        nn_reg.save(os.path.join(NOWCASTING_MODELS_DIR, 'nn_regressor.keras'))
    if nn_clf is not None:
        nn_clf.save(os.path.join(NOWCASTING_MODELS_DIR, 'nn_classifier.keras'))

    # Save test set predictions for metrics
    test_predictions = {
        'y_true_reg': y_test_reg.tolist(),
        'y_true_clf': y_test_clf.tolist(),
        'ridge_pred': ridge_pred.tolist(),
        'rf_pred': rf_pred.tolist(),
        'logreg_proba': logreg_proba.tolist(),
        'rf_clf_proba': rf_clf_proba.tolist(),
        'logreg_pred': logreg_pred.tolist(),
        'rf_clf_pred': rf_clf_pred.tolist(),
        'naive_lag1': df_test['naive_lag1'].tolist(),
        'routes': df_test['route'].tolist(),
        'airline_routes': df_test['airline_route'].tolist(),
        'year_month': df_test['year_month'].tolist(),
        'airlines': df_test['airline'].tolist(),
    }
    if xgb_clf is not None:
        test_predictions['xgb_proba'] = xgb_proba.tolist()
        test_predictions['xgb_pred'] = xgb_pred.tolist()
    if nn_reg_pred is not None:
        test_predictions['nn_reg_pred'] = nn_reg_pred.tolist()
    if nn_clf_proba is not None:
        test_predictions['nn_clf_proba'] = nn_clf_proba.tolist()
        test_predictions['nn_clf_pred'] = nn_clf_pred.tolist()

    with open(os.path.join(NOWCASTING_MODELS_DIR, 'test_predictions.json'), 'w') as f:
        json.dump(test_predictions, f)

    # Save full dataset predictions for time series visualisation
    print("\nGenerating predictions for full dataset (train+val+test)...")
    X_all = df_processed[feature_names].values
    X_all_scaled = scaler.transform(X_all)

    ridge_pred_all = ridge.predict(X_all_scaled)
    rf_pred_all = rf_reg.predict(X_all)
    nn_reg_pred_all = nn_reg.predict(X_all_scaled, verbose=0).flatten() if nn_reg is not None else None

    full_predictions = {
        'y_true_reg': df_processed['delay_rate'].tolist(),
        'ridge_pred': ridge_pred_all.tolist(),
        'rf_pred': rf_pred_all.tolist(),
        'year_month': df_processed['year_month'].tolist(),
        'year': df_processed['year'].tolist(),
        'split': [],  # 'train', 'val', or 'test'
    }

    # Mark which split each row belongs to
    for i in df_processed.index:
        if splits['train_mask'][i]:
            full_predictions['split'].append('train')
        elif splits['val_mask'][i]:
            full_predictions['split'].append('val')
        else:
            full_predictions['split'].append('test')

    if nn_reg_pred_all is not None:
        full_predictions['nn_reg_pred'] = nn_reg_pred_all.tolist()

    with open(os.path.join(NOWCASTING_MODELS_DIR, 'full_predictions.json'), 'w') as f:
        json.dump(full_predictions, f)

    # Save metadata
    metadata = {
        'feature_names': feature_names,
        'airline_cols': airline_cols,
        'route_cols': route_cols,
        'max_values': {k: float(v) for k, v in max_values.items()},
        'metrics': {k: {mk: float(mv) for mk, mv in v.items()} for k, v in metrics.items()},
        'route_metrics': route_metrics,
        'feature_importance': importance,
        'baseline_lag1_r2': overall_lag1_r2,
        'train_mean': train_mean,
        'split': {
            'train': 'years 2010-2017 + 2023',
            'val': 'years 2018 + 2024',
            'test': 'years 2019 + 2025',
            'n_train': int(X_train.shape[0]),
            'n_val': int(X_val.shape[0]),
            'n_test': int(X_test.shape[0]),
        },
        'valid_airlines': sorted(df_processed['airline'].unique().tolist()),
        'valid_routes': sorted(df_processed['route'].unique().tolist()),
        'valid_airline_routes': sorted(df_processed['airline_route'].unique().tolist()),
    }

    with open(os.path.join(NOWCASTING_MODELS_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll artifacts saved to {NOWCASTING_MODELS_DIR}/")
    print("Done.")

## Forecasting models ##
# Strictly using data prior to the prediction month
def train_and_save_forecasting():
    """Train forecasting models (only prior months' data, with LF_exp) and save to models/forecasting/."""

    # --- Load data ---
    print("Loading training data...")
    df = load_training_data()
    print(f"  Shape: {df.shape}")

    # --- Load factor data (from BITRE Monthly Airline Performance) ---
    print("\nLoading load factor data...")
    try:
        df_load_factor = load_load_factor_data()
        print(f"  Load factor records: {len(df_load_factor)}, "
              f"{df_load_factor['year_month'].min()} to {df_load_factor['year_month'].max()}")
    except FileNotFoundError as e:
        print(f"  Warning: {e}")
        print("  Continuing without load factor feature.")
        df_load_factor = None

    # --- Feature engineering (forecasting) ---
    print("\nRunning forecasting feature engineering pipeline...")
    result = build_forecasting_feature_matrix(df, df_load_factor=df_load_factor)
    df_processed = result['df']
    feature_names = result['feature_names']
    airline_cols = result['airline_cols']
    route_cols = result['route_cols']
    max_values = result['max_values']

    print(f"  Processed rows: {len(df_processed)}")
    print(f"  Features ({len(feature_names)}): {feature_names[-5:]} ...")
    using_lf = 'load_factor_lag1_exp' in feature_names
    print(f"  Load factor feature included: {using_lf}")

    # --- Split ---
    print("\nSplitting data...")
    splits = split_data(df_processed, feature_names)
    print(f"  Train: {splits['X_train'].shape[0]}")
    print(f"  Val:   {splits['X_val'].shape[0]}")
    print(f"  Test:  {splits['X_test'].shape[0]}")

    X_train = splits['X_train']
    X_val = splits['X_val']
    X_test = splits['X_test']
    y_train_reg = splits['y_train_reg']
    y_val_reg = splits['y_val_reg']
    y_test_reg = splits['y_test_reg']
    y_train_clf = splits['y_train_clf']
    y_val_clf = splits['y_val_clf']
    y_test_clf = splits['y_test_clf']

    # --- Scale ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # --- Train models ---
    metrics = {}

    # Ridge Regression (alpha=100 per notebook 15a)
    print("\nTraining Ridge Regression (forecasting)...")
    ridge = Ridge(alpha=100)
    ridge.fit(X_train_scaled, y_train_reg)
    ridge_pred = ridge.predict(X_test_scaled)
    ridge_r2 = r2_score(y_test_reg, ridge_pred)
    ridge_rmse = np.sqrt(mean_squared_error(y_test_reg, ridge_pred))
    ridge_mae = mean_absolute_error(y_test_reg, ridge_pred)
    print(f"  Test R\u00b2: {ridge_r2:.4f}, RMSE: {ridge_rmse:.4f}, MAE: {ridge_mae:.4f}")
    metrics['ridge'] = {'R2': ridge_r2, 'RMSE': ridge_rmse, 'MAE': ridge_mae}

    # Random Forest Regression (unscaled)
    print("Training Random Forest Regression (forecasting)...")
    rf_reg = RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    rf_reg.fit(X_train, y_train_reg)
    rf_pred = rf_reg.predict(X_test)
    rf_r2 = r2_score(y_test_reg, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test_reg, rf_pred))
    rf_mae = mean_absolute_error(y_test_reg, rf_pred)
    print(f"  Test R\u00b2: {rf_r2:.4f}, RMSE: {rf_rmse:.4f}, MAE: {rf_mae:.4f}")
    metrics['rf_reg'] = {'R2': rf_r2, 'RMSE': rf_rmse, 'MAE': rf_mae}

    # Logistic Regression (scaled)
    print("Training Logistic Regression (forecasting)...")
    logreg = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    logreg.fit(X_train_scaled, y_train_clf)
    logreg_pred = logreg.predict(X_test_scaled)
    logreg_proba = logreg.predict_proba(X_test_scaled)[:, 1]
    logreg_f1 = f1_score(y_test_clf, logreg_pred)
    logreg_auc = roc_auc_score(y_test_clf, logreg_proba)
    logreg_precision = precision_score(y_test_clf, logreg_pred)
    logreg_recall = recall_score(y_test_clf, logreg_pred)
    print(f"  Test F1: {logreg_f1:.4f}, AUC: {logreg_auc:.4f}")
    metrics['logreg'] = {
        'F1': logreg_f1, 'AUC': logreg_auc,
        'Precision': logreg_precision, 'Recall': logreg_recall,
    }

    # Random Forest Classification (unscaled)
    print("Training Random Forest Classification (forecasting)...")
    rf_clf = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    rf_clf.fit(X_train, y_train_clf)
    rf_clf_pred = rf_clf.predict(X_test)
    rf_clf_proba = rf_clf.predict_proba(X_test)[:, 1]
    rf_clf_f1 = f1_score(y_test_clf, rf_clf_pred)
    rf_clf_auc = roc_auc_score(y_test_clf, rf_clf_proba)
    rf_clf_precision = precision_score(y_test_clf, rf_clf_pred)
    rf_clf_recall = recall_score(y_test_clf, rf_clf_pred)
    print(f"  Test F1: {rf_clf_f1:.4f}, AUC: {rf_clf_auc:.4f}")
    metrics['rf_clf'] = {
        'F1': rf_clf_f1, 'AUC': rf_clf_auc,
        'Precision': rf_clf_precision, 'Recall': rf_clf_recall,
    }

    # XGBoost Classification (unscaled)
    xgb_clf = None
    if HAS_XGB:
        print("Training XGBoost Classification (forecasting)...")
        xgb_clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            min_child_weight=5, random_state=42, n_jobs=-1,
        )
        xgb_clf.fit(X_train, y_train_clf, eval_set=[(X_val, y_val_clf)], verbose=False)
        xgb_pred = xgb_clf.predict(X_test)
        xgb_proba = xgb_clf.predict_proba(X_test)[:, 1]
        xgb_f1 = f1_score(y_test_clf, xgb_pred)
        xgb_auc = roc_auc_score(y_test_clf, xgb_proba)
        xgb_precision = precision_score(y_test_clf, xgb_pred)
        xgb_recall = recall_score(y_test_clf, xgb_pred)
        print(f"  Test F1: {xgb_f1:.4f}, AUC: {xgb_auc:.4f}")
        metrics['xgb_clf'] = {
            'F1': xgb_f1, 'AUC': xgb_auc,
            'Precision': xgb_precision, 'Recall': xgb_recall,
        }
    else:
        print("XGBoost not available, skipping.")

    # Neural Network (TensorFlow/Keras) — same architecture as nowcasting (32→16→1)
    nn_reg = None
    nn_clf = None
    nn_reg_pred = None
    nn_clf_proba = None
    nn_clf_pred = None

    if HAS_TF:
        tf.random.set_seed(42)
        np.random.seed(42)

        class EpochProgressCallback(keras.callbacks.Callback):
            def __init__(self, model_name, total_epochs):
                super().__init__()
                self.model_name = model_name
                self.total_epochs = total_epochs

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                val_loss = logs.get('val_loss', 0)
                if (epoch + 1) % 10 == 0:
                    print(f"  {self.model_name} Epoch {epoch + 1}/{self.total_epochs} "
                          f"- val_loss: {val_loss:.4f}")

        # Neural Network Regression (forecasting)
        print("\nTraining Neural Network Regression (forecasting)...")
        nn_reg = keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='linear'),
        ])
        nn_reg.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                       loss='mse', metrics=['mae'])
        early_stop_reg = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True, verbose=0
        )
        nn_reg.fit(
            X_train_scaled, y_train_reg,
            validation_data=(X_val_scaled, y_val_reg),
            epochs=500, batch_size=64,
            callbacks=[early_stop_reg, EpochProgressCallback("NN Reg (fcst)", 500)],
            verbose=0
        )
        nn_reg_pred = nn_reg.predict(X_test_scaled, verbose=0).flatten()
        nn_reg_r2 = r2_score(y_test_reg, nn_reg_pred)
        nn_reg_rmse = np.sqrt(mean_squared_error(y_test_reg, nn_reg_pred))
        nn_reg_mae = mean_absolute_error(y_test_reg, nn_reg_pred)
        print(f"  Test R\u00b2: {nn_reg_r2:.4f}, RMSE: {nn_reg_rmse:.4f}, MAE: {nn_reg_mae:.4f}")
        metrics['nn_reg'] = {'R2': nn_reg_r2, 'RMSE': nn_reg_rmse, 'MAE': nn_reg_mae}

        # Neural Network Classification (forecasting)
        tf.random.set_seed(42)
        np.random.seed(42)
        print("Training Neural Network Classification (forecasting)...")
        nn_clf = keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid'),
        ])
        nn_clf.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                       loss='binary_crossentropy', metrics=['accuracy'])
        early_stop_clf = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True, verbose=0
        )
        n_neg_fcst = int((y_train_clf == 0).sum())
        n_pos_fcst = int((y_train_clf == 1).sum())
        class_weight_fcst = {0: 1.0, 1: n_neg_fcst / max(n_pos_fcst, 1)}
        print(f"  Class weights: {class_weight_fcst}")
        nn_clf.fit(
            X_train_scaled, y_train_clf.astype(np.float32),
            validation_data=(X_val_scaled, y_val_clf.astype(np.float32)),
            epochs=500, batch_size=64,
            callbacks=[early_stop_clf, EpochProgressCallback("NN Clf (fcst)", 500)],
            class_weight=class_weight_fcst,
            verbose=0
        )
        nn_clf_proba = nn_clf.predict(X_test_scaled, verbose=0).flatten()
        nn_clf_pred = (nn_clf_proba >= 0.5).astype(int)
        nn_clf_f1 = f1_score(y_test_clf, nn_clf_pred)
        nn_clf_auc = roc_auc_score(y_test_clf, nn_clf_proba)
        nn_clf_precision = precision_score(y_test_clf, nn_clf_pred)
        nn_clf_recall = recall_score(y_test_clf, nn_clf_pred)
        print(f"  Test F1: {nn_clf_f1:.4f}, AUC: {nn_clf_auc:.4f}")
        metrics['nn_clf'] = {
            'F1': nn_clf_f1, 'AUC': nn_clf_auc,
            'Precision': nn_clf_precision, 'Recall': nn_clf_recall,
        }
    else:
        print("TensorFlow not available, skipping neural network training.")

    # --- Compute route-level performance ---
    df_test = df_processed[splits['test_mask']].copy()
    df_test['ridge_pred'] = ridge.predict(X_test_scaled)
    df_test['rf_pred'] = rf_reg.predict(X_test)
    df_test['naive_lag1'] = df_test['delay_rate_lag1']
    if nn_reg_pred is not None:
        df_test['nn_pred'] = nn_reg_pred

    route_metrics = {}
    for route in sorted(df_test['route'].unique()):
        rd = df_test[df_test['route'] == route]
        route_metrics[route] = {
            'ridge_r2': float(r2_score(rd['delay_rate'], rd['ridge_pred'])),
            'rf_r2': float(r2_score(rd['delay_rate'], rd['rf_pred'])),
            'lag1_r2': float(r2_score(rd['delay_rate'], rd['naive_lag1'])),
            'n_samples': int(len(rd)),
        }
        if nn_reg_pred is not None:
            route_metrics[route]['nn_r2'] = float(r2_score(rd['delay_rate'], rd['nn_pred']))

    overall_lag1_r2 = float(r2_score(y_test_reg, df_test['naive_lag1'].values))
    train_mean = float(y_train_reg.mean())

    importance = {
        name: float(imp)
        for name, imp in zip(feature_names, rf_reg.feature_importances_)
    }

    # --- Save artifacts ---
    save_dir = FORECASTING_MODELS_DIR
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nSaving forecasting models to {save_dir}/...")
    joblib.dump(ridge, os.path.join(save_dir, 'ridge_regressor.pkl'))
    joblib.dump(rf_reg, os.path.join(save_dir, 'rf_regressor.pkl'))
    joblib.dump(logreg, os.path.join(save_dir, 'logreg_classifier.pkl'))
    joblib.dump(rf_clf, os.path.join(save_dir, 'rf_classifier.pkl'))
    if xgb_clf is not None:
        joblib.dump(xgb_clf, os.path.join(save_dir, 'xgb_classifier.pkl'))
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
    if nn_reg is not None:
        nn_reg.save(os.path.join(save_dir, 'nn_regressor.keras'))
    if nn_clf is not None:
        nn_clf.save(os.path.join(save_dir, 'nn_classifier.keras'))

    # Save test predictions
    test_predictions = {
        'y_true_reg': y_test_reg.tolist(),
        'y_true_clf': y_test_clf.tolist(),
        'ridge_pred': ridge_pred.tolist(),
        'rf_pred': rf_pred.tolist(),
        'logreg_proba': logreg_proba.tolist(),
        'rf_clf_proba': rf_clf_proba.tolist(),
        'logreg_pred': logreg_pred.tolist(),
        'rf_clf_pred': rf_clf_pred.tolist(),
        'naive_lag1': df_test['naive_lag1'].tolist(),
        'routes': df_test['route'].tolist(),
    }
    if xgb_clf is not None:
        test_predictions['xgb_proba'] = xgb_proba.tolist()
        test_predictions['xgb_pred'] = xgb_pred.tolist()
    if nn_reg_pred is not None:
        test_predictions['nn_reg_pred'] = nn_reg_pred.tolist()
    if nn_clf_proba is not None:
        test_predictions['nn_clf_proba'] = nn_clf_proba.tolist()
        test_predictions['nn_clf_pred'] = nn_clf_pred.tolist()

    with open(os.path.join(save_dir, 'test_predictions.json'), 'w') as f:
        json.dump(test_predictions, f)

    # Save metadata
    metadata = {
        'feature_names': feature_names,
        'airline_cols': airline_cols,
        'route_cols': route_cols,
        'max_values': {k: float(v) for k, v in max_values.items()},
        'metrics': {k: {mk: float(mv) for mk, mv in v.items()} for k, v in metrics.items()},
        'route_metrics': route_metrics,
        'feature_importance': importance,
        'baseline_lag1_r2': overall_lag1_r2,
        'train_mean': train_mean,
        'uses_load_factor': using_lf,
        'split': {
            'train': 'years 2010-2017 + 2023',
            'val': 'years 2018 + 2024',
            'test': 'years 2019 + 2025',
            'n_train': int(X_train.shape[0]),
            'n_val': int(X_val.shape[0]),
            'n_test': int(X_test.shape[0]),
        },
        'valid_airlines': sorted(df_processed['airline'].unique().tolist()),
        'valid_routes': sorted(df_processed['route'].unique().tolist()),
        'valid_airline_routes': sorted(df_processed['airline_route'].unique().tolist()),
    }

    with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll forecasting artifacts saved to {save_dir}/")
    print("Done.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train FLAPS models')
    parser.add_argument('--forecasting', action='store_true',
                        help='Train forecasting models (no weather, with lag12)')
    args = parser.parse_args()

    if args.forecasting:
        train_and_save_forecasting()
    else:
        train_and_save()
