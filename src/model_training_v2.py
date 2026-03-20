"""
Model Training (V2) - Predictive Maintenance
============================================
Goal: Validate the redesigned feature engineering (V2) pipeline against 
      the 48h predictive horizon and compare with 24h performance.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = r"d:\HackMasters"
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "master_v2.csv")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "plots")

def train_and_evaluate(X_train, y_train, X_test, y_test, horizon_name):
    print(f"\n  Evaluating {horizon_name} horizon...")
    
    # Handle Class Imbalance
    scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_prob)
    print(f"    ROC-AUC: {auc:.4f}")
    print("    Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, auc

def main():
    print("\n" + "=" * 70)
    print("  MODEL VALIDATION (V2 REDESIGN)")
    print("=" * 70)

    df = pd.read_csv(INPUT_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["machineID", "datetime"]).reset_index(drop=True)

    # Exclude leakage and non-feature columns
    drop_cols = ["machineID", "datetime", "failure", "failed", "component_at_risk",
                 "failure_comp1", "failure_comp2", "failure_comp3", "failure_comp4",
                 "model", "age_group", "risk_level", "failure_within_24h", "failure_within_48h"]
    
    features = [c for c in df.columns if c not in drop_cols]
    
    # Categorical handling
    X = df[features].copy()
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = X[c].astype("category").cat.codes

    # Time-series split (80/20)
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    
    # 1. Compare 24h Horizon (Original)
    y24_train, y24_test = df["failure_within_24h"].iloc[:train_size], df["failure_within_24h"].iloc[train_size:]
    m24, auc24 = train_and_evaluate(X_train, y24_train, X_test, y24_test, "24-Hour")

    # 2. Compare 48h Horizon (New Goal)
    y48_train, y48_test = df["failure_within_48h"].iloc[:train_size], df["failure_within_48h"].iloc[train_size:]
    m48, auc48 = train_and_evaluate(X_train, y48_train, X_test, y48_test, "48-Hour")

    print("\n" + "=" * 70)
    print("  SUMMARY OF IMPROVEMENTS")
    print("=" * 70)
    print(f"  V2 Baseline (24h) AUC: {auc24:.4f}")
    print(f"  V2 Goal     (48h) AUC: {auc48:.4f}")
    
    print("\n  Top 10 Features (48h Model):")
    imp = pd.Series(m48.feature_importances_, index=features).sort_values(ascending=False)
    print(imp.head(10).to_string())

    # Save feature importance plot
    plt.figure(figsize=(10, 6))
    imp.head(15).plot(kind="barh", color='skyblue')
    plt.title("Top 15 Features for 48h Prediction (V2)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "v2_feature_importance_48h.png"), dpi=120)
    plt.close()

    # Save the 48h model for production inference
    model_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(model_dir, exist_ok=True)
    m48.save_model(os.path.join(model_dir, "xgboost_v2_48h.json"))
    print("  [SUCCESS] 48-Hour Model saved -> models/xgboost_v2_48h.json")

if __name__ == "__main__":
    main()
