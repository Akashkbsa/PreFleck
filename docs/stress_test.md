# Advanced Stress Testing - Failure Prediction Model

---

## TEST 1 -- Feature Ablation Test

Remove dominant features (`error_count_24h`, `iso_score`) and retrain.

```python
X_ablated = X_train.drop(columns=['error_count_24h', 'iso_score'])
X_test_ablated = X_test.drop(columns=['error_count_24h', 'iso_score'])

model_ablated = xgb.XGBClassifier(n_estimators=100, max_depth=5, scale_pos_weight=scale_pos_weight)
model_ablated.fit(X_ablated, y_train)

y_pred = model_ablated.predict(X_test_ablated)
print(classification_report(y_test, y_pred))
```

---

## TEST 2 -- Anomaly Score Leakage Test

Train Isolation Forest ONLY on training data to prevent leakage.

```python
iso_forest_leak_free = IsolationForest(contamination=0.02, random_state=42)
iso_forest_leak_free.fit(X_train_sensors)

df['iso_score_clean'] = iso_forest_leak_free.decision_function(X_sensors)
```

---

## TEST 3 -- Prediction Horizon Test

Shift failure labels forward (48h/72h) to test early warning capacity.

```python
def create_horizon_labels(df, hours):
    labels = pd.Series(0, index=df.index)
    failure_times = df[df['failed'] == 1]['datetime']
    for ft in failure_times:
        window_start = ft - pd.Timedelta(hours=hours)
        labels[(df['datetime'] >= window_start) & (df['datetime'] < ft)] = 1
    return labels

y_48h = create_horizon_labels(df, 48)
```

---

## TEST 4 -- Time Delay Simulation

Shift all features backward to simulate real-world data latency.

```python
df_delayed = df.groupby('machineID').shift(2) # 2-hour delay
```

---

## TEST 5 -- Noise Injection Test

Inject Gaussian noise to test model sensitivity to sensor variance.

```python
noise = np.random.normal(0, X_train.std() * 0.05, X_train.shape)
X_train_noisy = X_train + noise
```

---

## TEST 6 -- Missing Data Simulation

Simulate sensor failures by dropping data and testing imputation robustness.

```python
X_missing = X_test.copy()
mask = np.random.rand(*X_missing.shape) < 0.10
X_missing[mask] = np.nan
X_imputed = X_missing.fillna(X_train.mean())
```

---

## TEST 7 -- Cold Start Test

Train on one set of machines and test on completely unseen machine IDs.

```python
machines = df['machineID'].unique()
train_ids = machines[:80]
test_ids = machines[80:]

X_train_cold = df[df['machineID'].isin(train_ids)]
X_test_cold = df[df['machineID'].isin(test_ids)]
```

---

## TEST 8 -- Feature Importance Stability

Check if top predictors remain consistent across multiple training runs.

```python
importances = []
for i in range(5):
    m = xgb.XGBClassifier(random_state=i).fit(X_train, y_train)
    importances.append(m.feature_importances_)
```

---

## TEST 9 -- Model Simplicity Test

Compare complex XGBoost performance against simple Logistic Regression.

```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000).fit(X_train_scaled, y_train)
```

---

## TEST 10 -- Extreme Edge Case Testing

Test model behavior in low-event or high-sensor-value scenarios.

```python
X_edge = X_test[X_test['error_count_24h'] == 0]
y_pred_edge = model.predict(X_edge)
```
