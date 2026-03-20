# Advanced Feature Engineering Redesign (V2)

---

## STEP 1 -- Feature Dependency Analysis (Audit Results)

**Dominant Shortcut Features Found:**
1. `error_count_72h` (Importance: 79.6%)
2. `iso_score` (Importance: 5.8%)
3. `hours_since_maint` (Importance: 5.2%)

**Verdict:** The model is "cheating" by focusing primarily on error counts which spike immediately before failure. To predict 48h+ in advance, we must weaken these signals.

---

## STEP 2 -- Remove / Weaken Shortcut Features

We will replace `error_count_72h` with a longer-term window (`error_count_168h` / 7 days) and a lagged version (`error_count_lag_24`) to force the model to look further back.

```python
# Weakening current indicators
df['error_count_long'] = df.groupby('machineID')['error_count_24h'].transform(lambda x: x.rolling(168, min_periods=1).sum())
```

---

## STEP 3 -- Create Interaction Features

Capture the combined physical stress on the machine.

```python
# physical Stress Interactions
df['vibration_pressure_ratio'] = df['vibration'] / (df['pressure'] + 1)
df['energy_consumption'] = df['volt'] * df['rotate']
df['stress_score'] = (df['vibration_rmean_24'] * df['pressure_rmean_24']) / (df['age'] + 1)
```

---

## STEP 4 -- Time-Series Trend Features

Capture the direction of sensor change using slopes and EMA.

```python
# EMA for smoother trend detection
df['volt_ema_24'] = df.groupby('machineID')['volt'].transform(lambda x: x.ewm(span=24).mean())
df['pressure_slope'] = df.groupby('machineID')['pressure_rmean_24'].diff()
```

---

## STEP 5 -- Rate of Change Features

Detect accelerating wear and tear.

```python
# First and Second Derivatives
df['vibration_velocity'] = df.groupby('machineID')['vibration'].diff()
df['vibration_acceleration'] = df.groupby('machineID')['vibration_velocity'].diff()
```

---

## STEP 6 -- Long-Term Memory Features

Capture cumulative degradation over days/weeks.

```python
# Cumulative Error Counts and Maintenance History
df['cum_errors'] = df.groupby('machineID')['error_count_24h'].cumsum()
df['maint_frequency_30d'] = df.groupby('machineID')['maint_any'].transform(lambda x: x.rolling(720).sum())
```

---

## STEP 7 -- Multi-Horizon Lag Features

Ensure the model sees state changes 24h, 48h, and 72h ago.

```python
# Lags represent past machine states
for lag in [24, 48, 72]:
    df[f'volt_lag_{lag}'] = df.groupby('machineID')['volt'].shift(lag)
```

---

## STEP 8 -- Feature Smoothing

Reduce sensor jitter to reveal underlying degradation trends.

```python
# Savitzky-Golay or Rolling Mean smoothing
df['vibration_smooth'] = df.groupby('machineID')['vibration'].transform(lambda x: x.rolling(12, center=True).mean())
```

---

## STEP 9 -- Feature Selection & Reduction

Keep only generalizable features using correlation and importance filtering.

```python
# Dropping redundant high-correlation features (>0.95)
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
```

---

## STEP 10 -- Validation for 48h Horizon

Retrain the model targeting `failure_within_48h` to measure true predictive power.
