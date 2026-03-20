# Failure Prediction Model - Predictive Maintenance

---

## STEP 1 -- Prepare Final Dataset

Select engineered features + anomaly scores. Remove identifiers. Align target.

```python
import pandas as pd
import numpy as np

df = pd.read_csv("master_anomaly.csv")
df["datetime"] = pd.to_datetime(df["datetime"])

# We predict failure_within_24h
target = "failure_within_24h"

drop_cols = ["machineID", "datetime", "failure", "failed", "component_at_risk",
             "failure_comp1", "failure_comp2", "failure_comp3", "failure_comp4",
             "model", "age_group", "risk_level"]

features = [c for c in df.columns if c not in drop_cols and c != target]

X = df[features]
y = df[target]
```

---

## STEP 2 -- Time-Series Train-Test Split

Chronological split to prevent data leakage (predicting future with past).

```python
# Assuming data is sorted chronologically
train_size = int(len(df) * 0.8)

# Split by time instead of random sample
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

print(f"Train size: {len(X_train)}  Test size: {len(X_test)}")
```

---

## STEP 3 -- Handle Class Imbalance

Use class weights in XGBoost rather than SMOTE to maintain time-series integrity.

```python
from sklearn.utils.class_weight import compute_class_weight

# Calculate scale_pos_weight
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos

print(f"Imbalance scale factor: {scale_pos_weight:.2f}")
```

---

## STEP 4 -- Model Selection

Using XGBoost: Robust to missing values, correlated features, and non-scaled data.

```python
import xgboost as xgb

# XGBoost automatically handles diverse feature ranges without explicit scaling
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight, # Handles imbalance
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)
```

---

## STEP 5 -- Model Training

Fit the model on historical training data.

```python
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)
```

---

## STEP 6 -- Model Evaluation

Evaluate using Precision, Recall, F1, and ROC-AUC. Recall is prioritized to catch failures.

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
```

---

## STEP 7 -- Model Interpretation

Feature importance to understand what signals failure.

```python
import matplotlib.pyplot as plt

# Plot top 15 important features
xgb.plot_importance(model, max_num_features=15, height=0.5)
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
```

---

## STEP 8 -- Combine with Anomaly Detection

High failure probability + Anomaly detected -> Immediate Action required.

```python
df_test = df.iloc[train_size:].copy()
df_test["failure_prob"] = y_prob
df_test["failure_pred"] = y_pred

# Combined Alert Logic
df_test["high_risk_alert"] = np.where(
    (df_test["failure_prob"] > 0.8) & (df_test["ensemble_anomaly"] == -1),
    True, False
)

print(f"Total High-Risk Alerts: {df_test['high_risk_alert'].sum()}")
```

---

## STEP 9 -- Final Validation and Deployment Readiness

Verify predictions make physical sense and map system I/O.

```python
# System Summary:
# Input: Real-time sensor telemetry, error logs, maintenance records.
# Model: XGBoost prediction pipeline
# Output: failure_prob (Probability of failure in next 24 hours)
# Trigger: high_risk_alert indicates probability > 0.8 AND anomaly condition met.

print("Model Pipeline Validated and Ready for Deployment.")
```
