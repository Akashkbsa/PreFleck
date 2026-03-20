# Anomaly Detection Pipeline - Predictive Maintenance

---

## STEP 1 -- Prepare the Feature Dataset

Select numeric features only, drop identifiers and labels for unsupervised training.

```python
import pandas as pd
import numpy as np

df = pd.read_csv("master_featured.csv")
df["datetime"] = pd.to_datetime(df["datetime"])

# Columns to exclude from feature matrix
DROP_COLS = ["datetime", "machineID", "failure", "failed",
             "failure_within_24h", "component_at_risk",
             "failure_comp1", "failure_comp2", "failure_comp3", "failure_comp4",
             "model", "age_group"]

feature_cols = [c for c in df.columns if c not in DROP_COLS
                and df[c].dtype in ["float64", "float32", "int64", "int32"]]

X = df[feature_cols].copy()
y = df["failure_within_24h"].copy()
```

---

## STEP 2 -- Feature Scaling

Standardize all features so no single sensor dominates distance calculations.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
```

---

## STEP 3A -- Isolation Forest

Isolate anomalies by random recursive splits; anomalies need fewer splits.

```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_scaled)

df["iso_score"]  = iso_forest.decision_function(X_scaled)
df["iso_anomaly"] = iso_forest.predict(X_scaled)  # 1=normal, -1=anomaly
```

---

## STEP 3B -- One-Class SVM

Learn a boundary around normal data; points outside are anomalies.

```python
from sklearn.svm import OneClassSVM

# Subsample for training (OC-SVM is memory-heavy)
sample_idx = X_scaled.sample(n=50000, random_state=42).index
ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.02)
ocsvm.fit(X_scaled.loc[sample_idx])

df["ocsvm_score"]   = ocsvm.decision_function(X_scaled)
df["ocsvm_anomaly"] = ocsvm.predict(X_scaled)
```

---

## STEP 3C -- Autoencoder Neural Network

Learn to compress and reconstruct normal data; high reconstruction error = anomaly.

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

input_dim = X_scaled.shape[1]

inp = Input(shape=(input_dim,))
encoded = Dense(64, activation="relu")(inp)
encoded = Dense(32, activation="relu")(encoded)
latent  = Dense(16, activation="relu")(encoded)
decoded = Dense(32, activation="relu")(latent)
decoded = Dense(64, activation="relu")(decoded)
output  = Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(inp, output)
autoencoder.compile(optimizer="adam", loss="mse")

# Train on normal data only
normal_mask = df["failure_within_24h"] == 0
autoencoder.fit(
    X_scaled[normal_mask], X_scaled[normal_mask],
    epochs=50, batch_size=256,
    validation_split=0.1,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=1
)

# Reconstruction error as anomaly score
reconstructed = autoencoder.predict(X_scaled)
df["ae_recon_error"] = np.mean((X_scaled.values - reconstructed) ** 2, axis=1)
ae_threshold = df.loc[normal_mask, "ae_recon_error"].quantile(0.98)
df["ae_anomaly"] = np.where(df["ae_recon_error"] > ae_threshold, -1, 1)
```

---

## STEP 4 -- Generate Anomaly Scores

Combine scores from all three models into a unified anomaly indicator.

```python
df["anomaly_votes"] = (
    (df["iso_anomaly"] == -1).astype(int) +
    (df["ocsvm_anomaly"] == -1).astype(int) +
    (df["ae_anomaly"] == -1).astype(int)
)

# Ensemble: anomaly if majority (>=2) of models agree
df["ensemble_anomaly"] = np.where(df["anomaly_votes"] >= 2, -1, 1)
```

---

## STEP 5 -- Evaluate Anomaly Detection Performance

Compare detected anomalies with actual failure windows.

```python
from sklearn.metrics import classification_report, roc_auc_score

# Convert: anomaly=-1 -> positive=1, normal=1 -> negative=0
y_pred = (df["ensemble_anomaly"] == -1).astype(int)
y_true = df["failure_within_24h"]

print(classification_report(y_true, y_pred, target_names=["Normal", "Pre-Failure"]))
print(f"ROC-AUC: {roc_auc_score(y_true, -df['iso_score']):.4f}")
```

---

## STEP 6 -- Visualize Anomalies

Plot sensor time series with anomaly points highlighted.

```python
import matplotlib.pyplot as plt

machine = df[df["machineID"] == 1].copy()

fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)
for ax, col in zip(axes, ["volt", "rotate", "pressure", "vibration"]):
    ax.plot(machine["datetime"], machine[col], linewidth=0.5, alpha=0.7)
    anoms = machine[machine["ensemble_anomaly"] == -1]
    ax.scatter(anoms["datetime"], anoms[col], c="red", s=8, label="Anomaly")
    fails = machine[machine["failed"] == 1]
    ax.axvline(x=fails["datetime"].values, color="black", linestyle="--", alpha=0.5)
    ax.set_ylabel(col)
    ax.legend(loc="upper right")
axes[0].set_title("Machine 1 - Sensor Readings with Anomalies")
plt.tight_layout()
plt.savefig("anomaly_visualization.png", dpi=120)
plt.close()
```

---

## STEP 7 -- Build Early Warning System

Rule-based alert logic combining anomaly signals with sensor trends.

```python
def assess_risk(row):
    score = 0
    if row["ensemble_anomaly"] == -1:
        score += 3
    if row["vibration_trend"] > 0:
        score += 1
    if row["pressure_trend"] < 0:
        score += 1
    if row["error_count_24h"] > 2:
        score += 2
    if row["hours_since_maint"] > 200:
        score += 1

    if score >= 5:
        return "CRITICAL"
    elif score >= 3:
        return "HIGH"
    elif score >= 1:
        return "MEDIUM"
    return "LOW"

df["risk_level"] = df.apply(assess_risk, axis=1)
print(df["risk_level"].value_counts())
```

---

## FINAL STEP -- Validate Anomaly Detection Pipeline

Verify anomaly rate, correlation with failures, and temporal alignment.

```python
# 1. Anomaly rate
anomaly_rate = (df["ensemble_anomaly"] == -1).mean() * 100
print(f"Anomaly rate: {anomaly_rate:.2f}%")

# 2. Anomaly rate inside failure windows vs outside
in_window  = df[df["failure_within_24h"] == 1]
out_window = df[df["failure_within_24h"] == 0]
print(f"Anomaly rate in pre-failure windows : {(in_window['ensemble_anomaly'] == -1).mean()*100:.2f}%")
print(f"Anomaly rate in normal operation    : {(out_window['ensemble_anomaly'] == -1).mean()*100:.2f}%")

# 3. Save final dataset
df.to_csv("master_anomaly.csv", index=False)
```
