"""
Machine Learning Audit Script - Predictive Maintenance
======================================================
Performs a systematic 11-step audit of the ML pipeline 
to detect data leakage, overfitting, and logic errors.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
import os
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

INPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "master_anomaly.csv")

def main():
    print("\n" + "#" * 70)
    print("  MACHINE LEARNING PIPELINE AUDIT")
    print("#" * 70)

    # Load data
    df = pd.read_csv(INPUT_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    target = "failure_within_24h"
    drop_cols = ["machineID", "datetime", "failure", "failed", "component_at_risk",
                 "failure_comp1", "failure_comp2", "failure_comp3", "failure_comp4",
                 "model", "age_group", "risk_level"]
    
    features = [c for c in df.columns if c not in drop_cols and c != target]
    X = df[features].copy()
    # Handle categorical for XGB/Logistic
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = X[c].astype("category").cat.codes
    
    y = df[target]
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # STEP 1: Verify Train-Test Split (Time-Series Check)
    print("\n  STEP 1 -- Time-Series Split Verification")
    train_max = df.iloc[train_size-1]["datetime"]
    test_min = df.iloc[train_size]["datetime"]
    print(f"    Train Max Date: {train_max}")
    print(f"    Test Min Date : {test_min}")
    if train_max < test_min:
        print("    Status: [OK] No chronological overlap.")
    else:
        print("    [ALERT] Timing overlap detected!")

    # STEP 2: Check for Target Leakage
    print("\n  STEP 2 -- Target Leakage Detection")
    corrs = X.corrwith(y).abs().sort_values(ascending=False)
    high_corrs = corrs[corrs > 0.8]
    if not high_corrs.empty:
        print("    [WARN] Highly correlated features (>0.8):")
        for f, v in high_corrs.items():
            print(f"      {f:40s} r={v:.4f}")
    else:
        print("    Status: [OK] No features with extreme correlation.")

    # STEP 3: Verify Feature Engineering Logic
    print("\n  STEP 3 -- Feature Engineering Logic Check")
    # Verify volt_lag_1 is correctly shifted
    idx = 500
    curr_volt = df.loc[idx, "volt"]
    next_lag = df.loc[idx+1, "volt_lag_1"]
    if curr_volt == next_lag:
        print("    Status: [OK] Lag-1 logic verified.")
    else:
        print("    [ALERT] Lag-1 logic mismatch!")

    # STEP 4: Check for Data Duplication
    print("\n  STEP 4 -- Data Duplication Check")
    dups = df.duplicated(subset=features).sum()
    print(f"    Duplicate feature rows: {dups}")
    if dups > 1000: # Some duplicates are normal in discrete sensors, but 800k is weird
        print("    [WARN] High number of duplicate rows.")
    else:
        print("    Status: [OK] Duplicate levels reasonable.")

    # STEP 5: Validate Class Distribution
    print("\n  STEP 5 -- Class Distribution Validation")
    ratio = y.value_counts(normalize=True)
    print(f"    Failure Ratio: {ratio.get(1, 0)*100:.2f}%")
    if ratio.get(1, 0) < 0.001:
        print("    [WARN] Extremely rare classes; metrics may be unstable.")

    # STEP 6: Simple Model Test (Leakage Detector)
    print("\n  STEP 6 -- Baseline (Logistic Regression) Leakage Test")
    lr = LogisticRegression(max_iter=100)
    lr.fit(X_train.fillna(0), y_train)
    lr_pred = lr.predict(X_test.fillna(0))
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test.fillna(0))[:,1])
    print(f"    Logistic Regression ROC-AUC: {lr_auc:.4f}")
    if lr_auc > 0.95:
        print("    [CAUTION] Simple LR is near-perfect; leakage likely exists.")

    # STEP 7: Time-Based Cross Validation
    print("\n  STEP 7 -- TimeSeriesSplit Validation")
    tscv = TimeSeriesSplit(n_splits=3)
    # Using small trees for speed
    clf = xgb.XGBClassifier(n_estimators=50, max_depth=3, n_jobs=-1, eval_metric="auc")
    scores = cross_val_score(clf, X, y, cv=tscv, scoring="roc_auc")
    print(f"    TS-CV ROC-AUC scores: {scores}")
    print(f"    Mean ROC-AUC: {np.mean(scores):.4f}")

    # STEP 8: Feature Importance Analysis
    print("\n  STEP 8 -- Critical Feature Inspection")
    clf.fit(X_train, y_train)
    imp = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
    print("    Top 5 Features:")
    print(imp.head(5).to_string())
    top_f = imp.index[0]
    if imp.iloc[0] > 0.5:
        print(f"    [WARN] Feature '{top_f}' accounts for >50% of model info.")

    # STEP 9: Overfitting Check
    print("\n  STEP 9 -- Overfitting (Bias-Variance) Check")
    train_pred = clf.predict_proba(X_train)[:,1]
    test_pred = clf.predict_proba(X_test)[:,1]
    train_auc = roc_auc_score(y_train, train_pred)
    test_auc = roc_auc_score(y_test, test_pred)
    print(f"    Train ROC-AUC: {train_auc:.4f}")
    print(f"    Test  ROC-AUC: {test_auc:.4f}")
    if (train_auc - test_auc) > 0.05:
        print("    [WARN] High variance detected (Overfitting).")

    # STEP 10: Anomaly Score Relation
    print("\n  STEP 10 -- Anomaly Detection Leakage Check")
    anom_corr = df["ensemble_anomaly"].corr(y)
    print(f"    Anomaly Score / Target Correlation: {anom_corr:.4f}")
    if abs(anom_corr) > 0.7:
        print("    [WARN] Anomaly scores are nearly identical to failure labels.")

    # STEP 11: Final Sanity Check (Shuffle Labels)
    print("\n  STEP 11 -- Label Shuffling (Null Hypothesis Test)")
    y_shuffled = y_train.sample(frac=1, random_state=42).values
    clf.fit(X_train, y_shuffled)
    shuff_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
    print(f"    Shuffled Target ROC-AUC: {shuff_auc:.4f}")
    if shuff_auc > 0.6:
        print("    [FAILED] Shuffled model still shows signal; leakage confirmed!")
    else:
        print("    Status: [OK] Shuffled labels result in random noise.")

    print("\n" + "#" * 70)
    print("  AUDIT COMPLETE")
    print("#" * 70 + "\n")

if __name__ == "__main__":
    main()
