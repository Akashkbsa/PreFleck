import pandas as pd
import xgboost as xgb
import os

PROJECT_ROOT = r"d:\HackMasters"
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "master_anomaly.csv")

df = pd.read_csv(INPUT_PATH)
target = "failure_within_24h"
drop_cols = ["machineID", "datetime", "failure", "failed", "component_at_risk", 
             "failure_comp1", "failure_comp2", "failure_comp3", "failure_comp4", 
             "model", "age_group", "risk_level", target]

features = [c for c in df.columns if c not in drop_cols]
X = df[features].copy()
for c in X.columns:
    if X[c].dtype == "object":
        X[c] = X[c].astype("category").cat.codes
y = df[target]

model = xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X, y)

print("\nTOP 10 FEATURES:")
imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print(imp.head(10))
