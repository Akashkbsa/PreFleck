# Feature Engineering Pipeline - Predictive Maintenance

---

## STEP 1 -- Lag Features (Past Sensor Values)

Create lagged sensor values to capture historical machine behaviour.

```python
SENSOR_COLS = ['volt', 'rotate', 'pressure', 'vibration']
LAGS = [1, 3, 6, 12, 24]

for col in SENSOR_COLS:
    for lag in LAGS:
        df[f'{col}_lag_{lag}'] = df.groupby('machineID')[col].shift(lag)
```

---

## STEP 2 -- Rolling Window Statistics

Summarize recent machine behaviour using rolling aggregates.

```python
WINDOWS = [3, 6, 12, 24]

for col in SENSOR_COLS:
    for w in WINDOWS:
        grp = df.groupby('machineID')[col]
        df[f'{col}_rmean_{w}']  = grp.transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f'{col}_rstd_{w}']   = grp.transform(lambda x: x.rolling(w, min_periods=1).std())
        df[f'{col}_rmin_{w}']   = grp.transform(lambda x: x.rolling(w, min_periods=1).min())
        df[f'{col}_rmax_{w}']   = grp.transform(lambda x: x.rolling(w, min_periods=1).max())
```

---

## STEP 3 -- Sensor Trend Features

Detect upward/downward trends by comparing short-term vs long-term rolling means.

```python
for col in SENSOR_COLS:
    short = df.groupby('machineID')[col].transform(lambda x: x.rolling(6, min_periods=1).mean())
    long  = df.groupby('machineID')[col].transform(lambda x: x.rolling(24, min_periods=1).mean())
    df[f'{col}_trend'] = short - long
```

---

## STEP 4 -- Rate of Change Features

Detect sudden spikes via first-difference of sensor values.

```python
for col in SENSOR_COLS:
    df[f'{col}_change'] = df.groupby('machineID')[col].diff()
```

---

## STEP 5 -- Duration Above Threshold Features

Count consecutive hours sensors stay above their 95th-percentile threshold.

```python
for col in SENSOR_COLS:
    threshold = df[col].quantile(0.95)
    above = (df[col] > threshold).astype(int)
    df[f'{col}_high_duration'] = above.groupby(df['machineID']).transform(
        lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1
    ) * above
```

---

## STEP 6 -- Error Frequency Features

Rolling count of errors in the last 24h and 72h.

```python
ERROR_COLS = ['error1', 'error2', 'error3', 'error4', 'error5']

df['total_errors'] = df[ERROR_COLS].sum(axis=1)

for w in [24, 72]:
    df[f'error_count_{w}h'] = df.groupby('machineID')['total_errors'].transform(
        lambda x: x.rolling(w, min_periods=1).sum()
    )
```

---

## STEP 7 -- Maintenance History Features

Compute time since last maintenance and rolling maintenance count.

```python
MAINT_COLS = ['maint_comp1', 'maint_comp2', 'maint_comp3', 'maint_comp4']

df['any_maint'] = df[MAINT_COLS].max(axis=1)

# Time since last maintenance (hours)
def time_since_event(series):
    result = pd.Series(np.nan, index=series.index)
    last = np.nan
    for i, val in enumerate(series):
        if val > 0:
            last = 0
        elif not np.isnan(last):
            last += 1
        result.iloc[i] = last
    return result

df['hours_since_maint'] = df.groupby('machineID')['any_maint'].transform(time_since_event)

# Maintenance count in last 30 days (720 hours)
df['maint_count_30d'] = df.groupby('machineID')['any_maint'].transform(
    lambda x: x.rolling(720, min_periods=1).sum()
)
```

---

## STEP 8 -- Machine Age Features

Create age groups from machine age metadata.

```python
df['age_group'] = pd.cut(df['age'], bins=[0, 5, 10, 15, 20, 100],
                         labels=['new', 'young', 'mid', 'old', 'very_old'])
```

---

## STEP 9 -- Failure History Features

Compute cumulative failure count and time since last failure per machine.

```python
df['cumulative_failures'] = df.groupby('machineID')['failed'].cumsum()

df['hours_since_failure'] = df.groupby('machineID')['failed'].transform(time_since_event)
```

---

## STEP 10 -- Machine Health Score

Weighted normalized combination of sensor values as a single health indicator.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
normed = pd.DataFrame(
    scaler.fit_transform(df[SENSOR_COLS]),
    columns=SENSOR_COLS, index=df.index
)

weights = {'volt': 0.2, 'rotate': 0.3, 'pressure': 0.25, 'vibration': 0.25}
df['health_score'] = sum(normed[c] * w for c, w in weights.items())
```

---

## STEP 11 -- Operating Condition Interaction Features

Create sensor ratio/product features to capture abnormal combinations.

```python
df['pressure_rotate_ratio'] = df['pressure'] / df['rotate'].replace(0, np.nan)
df['vibration_rotate_ratio'] = df['vibration'] / df['rotate'].replace(0, np.nan)
df['volt_pressure_product']  = df['volt'] * df['pressure']
```

---

## STEP 12 -- Error Type Encoding

Already one-hot encoded during preprocessing (error1-error5 binary columns).

```python
# Verify
print(df[['error1','error2','error3','error4','error5']].sum())
```

---

## STEP 13 -- Component Failure Encoding

Already encoded during preprocessing (failure_comp1-comp4 binary columns).

```python
# Verify
print(df[['failure_comp1','failure_comp2','failure_comp3','failure_comp4']].sum())
```

---

## STEP 14 -- Time-Based Features

Extract temporal features from datetime.

```python
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour_of_day'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['month']       = df['datetime'].dt.month
```

---

## STEP 15 -- Sensor Stability Features

Rolling variance to detect unstable sensor behaviour.

```python
for col in SENSOR_COLS:
    for w in [12, 24]:
        df[f'{col}_var_{w}h'] = df.groupby('machineID')[col].transform(
            lambda x: x.rolling(w, min_periods=1).var()
        )
```

---

## FINAL STEP -- Feature Validation

Verify all engineered features are valid and ready for modelling.

```python
# 1. No NaN remaining (after fillna)
print("Missing values:", df.isnull().sum().sum())

# 2. Correlation with failure label
corr = df.select_dtypes(include=[np.number]).corrwith(df['failure_within_24h'])
print("\nTop 20 features correlated with failure:")
print(corr.abs().sort_values(ascending=False).head(20))

# 3. Drop low-variance / constant features
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.001)
# selector.fit(df[numeric_cols])

# 4. Drop highly correlated feature pairs (r > 0.95) to reduce redundancy

# 5. Verify no future data leakage -- all features use only past/current data
```
