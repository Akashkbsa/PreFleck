"""
Advanced Stress Testing Script - Predictive Maintenance
========================================================
Implements a 10-step stress test suite for an XGBoost failure model.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import os
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

INPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "master_anomaly.csv")

def evaluate_test(model, X, y, test_name):
    y_prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob)
    print(f"    {test_name:30s} ROC-AUC: {auc:.4f}")
    return auc

def main():
    print("\n" + "=" * 70)
    print("  ADVANCED STRESS TESTING SUITE")
    print("=" * 70)

    # Load and prepare data
    df = pd.read_csv(INPUT_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    target = "failure_within_24h"
    drop_cols = ["machineID", "datetime", "failure", "failed", "component_at_risk",
                 "failure_comp1", "failure_comp2", "failure_comp3", "failure_comp4",
                 "model", "age_group", "risk_level"]
    
    features = [c for c in df.columns if c not in drop_cols and c != target]
    X = df[features].copy()
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = X[c].astype("category").cat.codes
    
    y = df[target]
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())

    # Original Baseline
    print("\n  BASELINE MODEL")
    base_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, scale_pos_weight=scale_pos_weight, eval_metric="auc", random_state=42, n_jobs=-1)
    base_model.fit(X_train, y_train)
    base_auc = evaluate_test(base_model, X_test, y_test, "Original Baseline")

    # TEST 1: Feature Ablation
    print("\n  TEST 1 -- Feature Ablation (Drop Top Predictors)")
    top_drop = ["error_count_24h", "iso_score"]
    X_ablated_train = X_train.drop(columns=top_drop)
    X_ablated_test = X_test.drop(columns=top_drop)
    model_t1 = xgb.XGBClassifier(n_estimators=100, scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1)
    model_t1.fit(X_ablated_train, y_train)
    t1_auc = evaluate_test(model_t1, X_ablated_test, y_test, "No 'error_count'/'iso_score'")

    # TEST 2: Anomaly Score Leakage Check
    print("\n  TEST 2 -- Clean Anomaly Scores (Train-Only IF)")
    sensor_cols = ["volt", "rotate", "pressure", "vibration"]
    iso = IsolationForest(contamination=0.02, random_state=42).fit(X_train[sensor_cols])
    X_clean_train = X_train.copy()
    X_clean_test = X_test.copy()
    X_clean_train["iso_score"] = iso.decision_function(X_train[sensor_cols])
    X_clean_test["iso_score"] = iso.decision_function(X_test[sensor_cols])
    model_t2 = xgb.XGBClassifier(n_estimators=100, scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1)
    model_t2.fit(X_clean_train, y_train)
    t2_auc = evaluate_test(model_t2, X_clean_test, y_test, "Clean Anomaly Scores")

    # TEST 3: Prediction Horizon (48h)
    print("\n  TEST 3 -- Prediction Horizon Shift (48h)")
    # Re-labeling logic
    def get_horizon_y(dataframe, hours):
        labels_h = pd.Series(0, index=dataframe.index)
        for _, row in dataframe[dataframe['failed'] == 1].iterrows():
            ws = row['datetime'] - pd.Timedelta(hours=hours)
            labels_h[(dataframe['datetime'] >= ws) & (dataframe['datetime'] < row['datetime'])] = 1
        return labels_h
    
    y_48h = get_horizon_y(df, 48)
    y_tr_48, y_te_48 = y_48h.iloc[:train_size], y_48h.iloc[train_size:]
    scale_48 = (y_tr_48 == 0).sum() / max(1, (y_tr_48 == 1).sum())
    model_t3 = xgb.XGBClassifier(n_estimators=100, scale_pos_weight=scale_48, random_state=42, n_jobs=-1)
    model_t3.fit(X_train, y_tr_48)
    t3_auc = evaluate_test(model_t3, X_test, y_te_48, "48h Prediction Horizon")

    # TEST 4: Time Delay Simulation (2h Delay)
    print("\n  TEST 4 -- Time Delay Simulation (2h)")
    X_delayed_test = X_test.groupby(df.iloc[train_size:]["machineID"]).shift(2).fillna(0)
    t4_auc = evaluate_test(base_model, X_delayed_test, y_test, "2h Data Latency")

    # TEST 5: Noise Injection
    print("\n  TEST 5 -- Noise Injection (5% Std Dev)")
    X_noisy_test = X_test + np.random.normal(0, X_test.std() * 0.05, X_test.shape)
    t5_auc = evaluate_test(base_model, X_noisy_test.fillna(0), y_test, "Gaussian Noise")

    # TEST 6: Missing Data Simulation (10% Missing)
    print("\n  TEST 6 -- Missing Data Simulation (10% Drop)")
    X_missing = X_test.copy()
    mask = np.random.rand(*X_missing.shape) < 0.10
    X_missing[mask] = np.nan
    X_imputed = X_missing.fillna(X_train.mean())
    t6_auc = evaluate_test(base_model, X_imputed, y_test, "10% Missing Features")

    # TEST 7: Cold Start (Unseen Machines)
    print("\n  TEST 7 -- Cold Start (Unseen Machines)")
    machines = df["machineID"].unique()
    train_m = machines[:80]
    test_m = machines[80:]
    X_tr_c, y_tr_c = X[df["machineID"].isin(train_m)], y[df["machineID"].isin(train_m)]
    X_te_c, y_te_c = X[df["machineID"].isin(test_m)], y[df["machineID"].isin(test_m)]
    model_t7 = xgb.XGBClassifier(n_estimators=100, scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1)
    model_t7.fit(X_tr_c, y_tr_c)
    t7_auc = evaluate_test(model_t7, X_te_c, y_te_c, "Generalization to New Machines")

    # TEST 8: Feature Importance Stability
    print("\n  TEST 8 -- Feature Importance Stability")
    m1 = base_model.feature_importances_
    m2 = xgb.XGBClassifier(random_state=123).fit(X_train, y_train).feature_importances_
    stab = np.corrcoef(m1, m2)[0,1]
    print(f"    Importance Stability (r): {stab:.4f}")

    # TEST 9: Model Simplicity (Decision Tree)
    print("\n  TEST 9 -- Model Simplicity (Decision Tree)")
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
    dt.fit(X_train.fillna(0), y_train)
    t9_auc = evaluate_test(dt, X_test.fillna(0), y_test, "Shallow Decision Tree")

    # TEST 10: Extreme Edge Case (Zero Errors)
    print("\n  TEST 10 -- Extreme Edge Case (Zero Errors)")
    mask_zero = (X_test["error_count_24h"] == 0)
    if mask_zero.any():
        t10_auc = evaluate_test(base_model, X_test[mask_zero], y_test[mask_zero], "Normal Operation Samples")
    else:
        print("    No zero-error samples found in test set.")
        t10_auc = 0


    print("\n" + "=" * 70)
    print("  STRESS TEST SUMMARY")
    print("=" * 70)
    results = [
        ("Baseline", base_auc),
        ("Ablation", t1_auc),
        ("Clean Anomaly", t2_auc),
        ("48h Horizon", t3_auc),
        ("Latency (2h)", t4_auc),
        ("Noise (5%)", t5_auc),
        ("Missing (10%)", t6_auc),
        ("Cold Start", t7_auc),
        ("Simplicity", t9_auc),
    ]
    for n, s in results:
        diff = s - base_auc
        print(f"  {n:20s}: {s:.4f}  ({'+' if diff >= 0 else ''}{diff:.4f})")

    print("\n  FINAL VERDICT:")
    if all(s > 0.85 for n, s in results if n != "Simplicity"):
        print("  [VERDICT] FULLY ROBUST - Model holds high performance under extreme noise and shifts.")
    elif all(s > 0.75 for n, s in results if n != "Simplicity"):
        print("  [VERDICT] MODERATELY ROBUST - Sensitive to latency, but structurally sound.")
    else:
        print("  [VERDICT] NEEDS IMPROVEMENT - High variance or proxy dependency detected.")
    
    print(f"  Confidence Score: {int(base_auc * 10)}/10")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
