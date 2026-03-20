import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import os

DATA_DIR = r"d:\HackMasters\Dataset"
INPUT_PATH = os.path.join(DATA_DIR, "master_anomaly.csv")

df = pd.read_csv(INPUT_PATH)
y = df["failure_within_24h"]

drop_cols = ["machineID", "datetime", "failure", "failed", "component_at_risk", 
             "failure_comp1", "failure_comp2", "failure_comp3", "failure_comp4", 
             "model", "age_group", "risk_level", "failure_within_24h"]

features = [c for c in df.columns if c not in drop_cols]
X = df[features].copy()
for c in X.columns:
    if X[c].dtype == "object":
        X[c] = X[c].astype("category").cat.codes

train_size = int(len(df) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Test 1: Real Model
clf = xgb.XGBClassifier(n_estimators=50, max_depth=3, n_jobs=-1, eval_metric="auc")
clf.fit(X_train, y_train)
real_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
print(f"REAL AUC: {real_auc:.4f}")

# Test 2: Shuffled Labels
y_shuffled = y_train.sample(frac=1, random_state=42).values
clf.fit(X_train, y_shuffled)
shuff_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
print(f"SHUFFLED AUC: {shuff_auc:.4f}")

# Test 3: Top Feature Correlation
corrs = X.corrwith(y).abs().sort_values(ascending=False)
print("\nTOP CORRELATIONS:")
print(corrs.head(5))
