"""
Failure Prediction Model Pipeline - Predictive Maintenance
==========================================================
Input  : master_anomaly.csv (from anomaly detection)
Output : Trained model evaluation and feature importance
Steps  : 9-step Supervised Learning Pipeline
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import os
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "plots")

INPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "master_anomaly.csv")

def main():
    print("\n" + "#" * 60)
    print("  FAILURE PREDICTION MODEL PIPELINE")
    print("#" * 60)

    # STEP 1
    print("\n  STEP 1 -- Prepare Final Dataset")
    df = pd.read_csv(INPUT_PATH)
    # Sort strictly chronologically
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    target = "failure_within_24h"
    drop_cols = ["machineID", "datetime", "failure", "failed", "component_at_risk",
                 "failure_comp1", "failure_comp2", "failure_comp3", "failure_comp4",
                 "model", "age_group", "risk_level"]
    
    features = [c for c in df.columns if c not in drop_cols and c != target]
    print(f"  Selected {len(features)} predictive features.")
    
    # Handle object/string columns if any remain
    X = df[features].copy()
    for c in X.columns:
        if X[c].dtype == "object" or str(X[c].dtype) == "category":
            X[c] = X[c].astype("category")

    y = df[target]

    # STEP 2
    print("\n  STEP 2 -- Time-Series Train-Test Split")
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    print(f"  Train: {X_train.shape[0]:,} rows (Past data)")
    print(f"  Test : {X_test.shape[0]:,} rows (Future data)")

    # STEP 3
    print("\n  STEP 3 -- Handle Class Imbalance")
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_w = neg / max(1, pos)
    print(f"  Train class distribution: 0={neg:,}, 1={pos:,}")
    print(f"  scale_pos_weight calculated: {scale_pos_w:.2f}")

    # STEP 4 & 5
    print("\n  STEP 4 & 5 -- Model Selection & Training (XGBoost)")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_w,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        enable_categorical=True
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    print("  [OK] Model training complete.")

    # STEP 6
    print("\n  STEP 6 -- Model Evaluation")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"    TN: {cm[0][0]:6d}  FP: {cm[0][1]:6d}")
    print(f"    FN: {cm[1][0]:6d}  TP: {cm[1][1]:6d}")
    
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"  ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    # STEP 7
    print("\n  STEP 7 -- Model Interpretation (Feature Importance)")
    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(model, max_num_features=15, height=0.5, ax=ax)
    plt.title("Top 15 Predictors of Failure")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "feature_importance.png"), dpi=120)
    plt.close()
    print("  [OK] Feature importance plot saved -> feature_importance.png")

    # STEP 8
    print("\n  STEP 8 -- Combine with Anomaly Detection")
    df_test = df.iloc[train_size:].copy()
    df_test["failure_prob"] = y_prob
    df_test["failure_pred"] = y_pred
    
    # Check if 'ensemble_anomaly' exists
    if "ensemble_anomaly" in df_test.columns:
        df_test["high_risk"] = np.where(
            (df_test["failure_prob"] > 0.8) & (df_test["ensemble_anomaly"] == -1),
            1, 0
        )
        print(f"  High-risk alerts triggered (Prob > 0.8 & Anomaly == -1): {df_test['high_risk'].sum():,}")

    # STEP 9
    print("\n  STEP 9 -- Final Validation & Deployment Readiness")
    print("  PIPELINE READY:")
    print("  - Input   : Live sensor stream, logs, & engineered features")
    print("  - Engine  : XGBoost + Ensemble Anomaly Detector")
    print("  - Output  : Probability of failure within 24h & Alert Trigger")

    print("\n" + "#" * 60)
    print("  SYSTEM COMPLETE")
    print("#" * 60 + "\n")

if __name__ == "__main__":
    main()
