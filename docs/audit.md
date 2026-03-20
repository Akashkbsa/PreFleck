# Machine Learning Model Audit - Predictive Maintenance

---

## STEP 1 -- Verify Train-Test Split (Time-Series Check)

Ensure chronological splitting and no future data leakage into training.

```python
train_max_date = df_train['datetime'].max()
test_min_date = df_test['datetime'].min()

if train_max_date < test_min_date:
    print("Chronological split: [OK]")
else:
    print("Leakage detected: Training data contains future timestamps!")
```

---

## STEP 2 -- Check for Target Leakage

Identify features with suspiciously high correlation to the target variable.

```python
correlations = df.corr()['failure_within_24h'].abs().sort_values(ascending=False)
potential_leakage = correlations[correlations > 0.95]
print("Potential Leakage Columns:", potential_leakage.index.tolist())
```

---

## STEP 3 -- Verify Feature Engineering Logic

Confirm lag and rolling features only use past data.

```python
# Check lag_1: current value at T should be previous value at T-1
val_t = df.loc[100, 'volt']
lag_t1 = df.loc[101, 'volt_lag_1']
if val_t == lag_t1:
    print("Lag Feature Logic: [OK]")
```

---

## STEP 4 -- Check for Data Duplication or Overlap

Verify no data points are shared between training and testing sets.

```python
duplicates = df.duplicated().sum()
overlap = pd.merge(df_train, df_test, how='inner').shape[0]
print(f"Duplicates: {duplicates}, Overlapping Rows: {overlap}")
```

---

## STEP 5 -- Validate Class Distribution

Ensure failure rates are realistic and not artificially inflated.

```python
print("Class Distribution:\n", df['failure_within_24h'].value_counts(normalize=True))
```

---

## STEP 6 -- Simple Model Test (Leakage Detector)

A Logistic Regression performing near-perfectly is a clear sign of leakage.

```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression().fit(X_train, y_train)
print("Logistic Regression Score:", lr.score(X_test, y_test))
```

---

## STEP 7 -- Time-Based Cross Validation

Use TimeSeriesSplit to validate model stability across different time windows.

```python
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv)
print("TS Cross-Val Scores:", scores)
```

---

## STEP 8 -- Feature Importance Analysis

Inspect the top predictors to ensure they are not direct proxies for failure.

```python
importances = model.feature_importances_
top_feature = features[np.argmax(importances)]
print("Top Feature:", top_feature)
```

---

## STEP 9 -- Overfitting Check

Compare train vs test metrics to detect high variance.

```python
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
```

---

## STEP 10 -- Anomaly + Failure Label Relation Check

Verify if anomaly detection was accidentally exposed to the target label.

```python
score_corr = df['ensemble_anomaly'].corr(df['failure_within_24h'])
print("Anomaly-Target Correlation:", score_corr)
```

---

## STEP 11 -- Final Sanity Checks

Retrain with shuffled labels; performance must drop to random-chance levels.

```python
y_shuffled = y.sample(frac=1).values
model.fit(X_train, y_shuffled[:len(X_train)])
print("Shuffled Label Score:", model.score(X_test, y_test[train_size:]))
```
