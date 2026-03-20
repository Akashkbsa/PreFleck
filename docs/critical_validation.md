# Critical Final Validation - Predictive Maintenance

---

## SECTION 1 -- Maintenance Feature Leakage Check

**Goal:** Ensure maintenance features (`maint_count_30d`, `hours_since_maint`) are not cheating proxies for failure.

```python
# 1. Temporal Analysis
failures_only = df[df["failed"] == 1]
mean_hours = failures_only["hours_since_maint"].mean()

# 2. Leakage Test: Retrain without maintenance features
X_no_maint = X_train.drop(columns=["maint_count_30d", "hours_since_maint"])
```

---

## SECTION 2 -- Label Alignment / Future Data Leakage Check

**Goal:** Ensure predicting at T uses strictly data ≤ T.

```python
# Feature Shift Check
X_shifted = X_test.groupby('machineID').shift(48).fillna(0)
auc_shifted = roc_auc_score(y_test, model.predict_proba(X_shifted))
# If AUC holds up without dropping, alignment is correct. 
# Alternatively, intentionally shifting features forward should destroy performance.
```

---

## SECTION 3 -- Rolling / EMA Future Leakage Check

**Goal:** Verify rolling and EMA calculations do not use future context.

```python
# Verified in implementation:
df['volt_ema_24'] = df['volt'].ewm(span=24).mean() # Default center=False
df['vibration_rmean_6'] = df['vibration'].rolling(6).mean() # Default center=False
```

---

## SECTION 4 -- Hard Time Cut Validation (Mandatory)

**Goal:** Split chronologically and verify robustness without spatial overlap.

```python
# Strict Date Split
split_date = pd.to_datetime("2015-10-15")
X_train = df[df["datetime"] < split_date]
X_test = df[df["datetime"] >= split_date]
```

---

## SECTION 5 -- Feature Sanity Drop Test (Critical)

**Goal:** Check if the model performs well solely on sensor signals (no maintenance, no anomaly scores).

```python
drop_sanity = ["iso_score", "hours_since_maint", "maint_count_30d"]
X_sanity = X_train.drop(columns=drop_sanity)
```
