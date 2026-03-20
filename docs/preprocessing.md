# Predictive Maintenance — Data Cleaning & Preprocessing Pipeline

---

## Imports & File Paths

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r"d:\HackMasters\Dataset"

telemetry = pd.read_csv(f"{DATA_DIR}/PdM_telemetry.csv")
errors    = pd.read_csv(f"{DATA_DIR}/PdM_errors.csv")
maint     = pd.read_csv(f"{DATA_DIR}/PdM_maint.csv")
failures  = pd.read_csv(f"{DATA_DIR}/PdM_failures.csv")
machines  = pd.read_csv(f"{DATA_DIR}/PdM_machines.csv")
```

---

## STAGE 1 — DATASET UNDERSTANDING

```python
datasets = {
    "telemetry": telemetry,
    "errors":    errors,
    "maint":     maint,
    "failures":  failures,
    "machines":  machines,
}

for name, df in datasets.items():
    print(f"\n{'='*60}")
    print(f"  {name.upper()}")
    print(f"{'='*60}")
    print(f"Shape : {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nFirst 5 rows:")
    display(df.head())
```

### Primary & Join Keys

```python
print("Unique machineIDs in machines:", machines['machineID'].nunique())

for name, df in datasets.items():
    if 'machineID' in df.columns:
        print(f"{name:12s}  → machineID range: "
              f"{df['machineID'].min()} – {df['machineID'].max()}, "
              f"unique: {df['machineID'].nunique()}")
```

### Time Coverage

```python
for name in ["telemetry", "errors", "maint", "failures"]:
    df = datasets[name]
    print(f"{name:12s}  → {df['datetime'].min()}  to  {df['datetime'].max()}")
```

### Machine Metadata

```python
print(machines['model'].value_counts())
print(f"\nAge statistics:\n{machines['age'].describe()}")
```

### Sensor Distributions

```python
sensor_cols = ['volt', 'rotate', 'pressure', 'vibration']
print(telemetry[sensor_cols].describe())

telemetry[sensor_cols].hist(bins=50, figsize=(14, 8))
plt.suptitle("Sensor Value Distributions (Raw)")
plt.tight_layout()
plt.show()
```

---

## STAGE 2 — DATA TYPE CORRECTION

### Datetime Conversion

```python
for name in ["telemetry", "errors", "maint", "failures"]:
    datasets[name]['datetime'] = pd.to_datetime(datasets[name]['datetime'])
    print(f"{name:12s}  dtype now: {datasets[name]['datetime'].dtype}")
```

### Verify Chronological Order

```python
for name in ["telemetry", "errors", "maint", "failures"]:
    df = datasets[name]
    diffs = df.groupby('machineID')['datetime'].diff()
    backward = (diffs < pd.Timedelta(0)).sum()
    print(f"{name:12s}  backward jumps: {backward}")
```

### Numeric Conversion

```python
sensor_cols = ['volt', 'rotate', 'pressure', 'vibration']
for col in sensor_cols:
    telemetry[col] = pd.to_numeric(telemetry[col], errors='coerce')

print(telemetry[sensor_cols].dtypes)
```

### Categorical Conversion

```python
machines['model']    = machines['model'].astype('category')
errors['errorID']    = errors['errorID'].astype('category')
maint['comp']        = maint['comp'].astype('category')
failures['failure']  = failures['failure'].astype('category')

for name, df in datasets.items():
    if 'machineID' in df.columns:
        df['machineID'] = df['machineID'].astype('category')
```

---

## STAGE 3 — TIME SERIES SORTING

```python
telemetry = telemetry.sort_values(['machineID', 'datetime']).reset_index(drop=True)
errors    = errors.sort_values(['machineID', 'datetime']).reset_index(drop=True)
maint     = maint.sort_values(['machineID', 'datetime']).reset_index(drop=True)
failures  = failures.sort_values(['machineID', 'datetime']).reset_index(drop=True)
```

### Verify Timestamp Consistency

```python
time_diffs = telemetry.groupby('machineID')['datetime'].diff()
non_hourly = time_diffs[time_diffs != pd.Timedelta(hours=1)]
print(f"Non-hourly gaps found: {non_hourly.dropna().shape[0]}")

if non_hourly.dropna().shape[0] > 0:
    print("\nSample of irregular gaps:")
    print(non_hourly.dropna().value_counts().head(10))
```

---

## STAGE 4 — MISSING VALUE ANALYSIS

### Detect Missing Values

```python
for name, df in datasets.items():
    total = df.isnull().sum()
    pct   = (df.isnull().sum() / len(df)) * 100
    missing = pd.DataFrame({'Missing': total, 'Percent': pct})
    missing = missing[missing['Missing'] > 0]
    if missing.empty:
        print(f"{name:12s}  → No missing values ✓")
    else:
        print(f"\n{name:12s}  → MISSING VALUES DETECTED:")
        print(missing)
```

### Pattern Analysis

```python
if telemetry[sensor_cols].isnull().any().any():
    machine_missing = telemetry.groupby('machineID')[sensor_cols].apply(
        lambda x: x.isnull().sum()
    )
    print("Missing values by machine:")
    print(machine_missing[machine_missing.sum(axis=1) > 0])

    telemetry['month'] = telemetry['datetime'].dt.to_period('M')
    month_missing = telemetry.groupby('month')[sensor_cols].apply(
        lambda x: x.isnull().sum()
    )
    print("\nMissing values by month:")
    print(month_missing[month_missing.sum(axis=1) > 0])
    telemetry.drop('month', axis=1, inplace=True)
```

### Imputation (Linear Interpolation → Forward Fill → Backward Fill)

```python
telemetry[sensor_cols] = telemetry.groupby('machineID')[sensor_cols].apply(
    lambda g: g.interpolate(method='linear')
)

telemetry[sensor_cols] = telemetry.groupby('machineID')[sensor_cols].ffill()

telemetry[sensor_cols] = telemetry.groupby('machineID')[sensor_cols].bfill()

assert telemetry[sensor_cols].isnull().sum().sum() == 0, "Still have NaNs!"
print("All missing sensor values imputed ✓")
```

---

## STAGE 5 — OUTLIER DETECTION

### Z-Score

```python
for col in sensor_cols:
    z = np.abs(stats.zscore(telemetry[col].dropna()))
    outliers = (z > 3).sum()
    print(f"{col:12s}  outliers (|z| > 3): {outliers} "
          f"({outliers / len(telemetry) * 100:.3f}%)")
```

### IQR

```python
for col in sensor_cols:
    Q1 = telemetry[col].quantile(0.25)
    Q3 = telemetry[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((telemetry[col] < lower) | (telemetry[col] > upper)).sum()
    print(f"{col:12s}  IQR outliers: {outliers} "
          f"(range: {lower:.2f} – {upper:.2f})")
```

### Visualization

```python
# Box Plots
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for i, col in enumerate(sensor_cols):
    telemetry.boxplot(column=col, ax=axes[i])
    axes[i].set_title(col)
plt.suptitle("Sensor Box Plots — Outlier Identification")
plt.tight_layout()
plt.show()

# Distribution Plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, col in zip(axes.flatten(), sensor_cols):
    sns.histplot(telemetry[col], bins=100, kde=True, ax=ax)
    ax.set_title(f"{col} Distribution")
plt.suptitle("Sensor Distributions")
plt.tight_layout()
plt.show()

# Time-Series Plot (Sample Machine)
sample_machine = 1
machine_data = telemetry[telemetry['machineID'] == sample_machine]

fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
for ax, col in zip(axes, sensor_cols):
    ax.plot(machine_data['datetime'], machine_data[col], linewidth=0.5)
    ax.set_ylabel(col)
axes[0].set_title(f"Machine {sample_machine} — Sensor Time Series")
plt.tight_layout()
plt.show()
```

### Domain-Specific Checks

```python
print("Voltage = 0       :", (telemetry['volt'] == 0).sum())
print("Negative pressure :", (telemetry['pressure'] < 0).sum())
print("Vibration > 100   :", (telemetry['vibration'] > 100).sum())
print("Rotation > 600    :", (telemetry['rotate'] > 600).sum())
```

### Winsorize (Cap at 1st / 99th Percentile)

```python
for col in sensor_cols:
    lower = telemetry[col].quantile(0.01)
    upper = telemetry[col].quantile(0.99)
    telemetry[col] = telemetry[col].clip(lower=lower, upper=upper)
    print(f"{col:12s}  clipped to [{lower:.2f}, {upper:.2f}]")
```

---

## STAGE 6 — SENSOR DATA SMOOTHING

### Rolling Mean (24h Window)

```python
WINDOW = 24

for col in sensor_cols:
    telemetry[f'{col}_rolling_mean'] = (
        telemetry.groupby('machineID')[col]
        .transform(lambda x: x.rolling(window=WINDOW, min_periods=1).mean())
    )

print("Rolling mean columns added ✓")
```

### Exponential Smoothing (EWM)

```python
SPAN = 24

for col in sensor_cols:
    telemetry[f'{col}_ewm'] = (
        telemetry.groupby('machineID')[col]
        .transform(lambda x: x.ewm(span=SPAN, min_periods=1).mean())
    )

print("EWM columns added ✓")
```

### Visual Comparison

```python
sample = telemetry[telemetry['machineID'] == 1].head(500)

fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
for ax, col in zip(axes, sensor_cols):
    ax.plot(sample['datetime'], sample[col], alpha=0.3, label='Raw')
    ax.plot(sample['datetime'], sample[f'{col}_rolling_mean'], label='Rolling Mean (24h)')
    ax.plot(sample['datetime'], sample[f'{col}_ewm'], label='EWM (span=24)')
    ax.set_ylabel(col)
    ax.legend(loc='upper right', fontsize=8)
axes[0].set_title("Machine 1 — Raw vs Smoothed Sensor Signals")
plt.tight_layout()
plt.show()
```

---

## STAGE 7 — DUPLICATE DATA HANDLING

### Detect Duplicates

```python
for name, df in datasets.items():
    if 'datetime' in df.columns:
        dup_count = df.duplicated(subset=['machineID', 'datetime']).sum()
        print(f"{name:12s}  exact duplicates on (machineID, datetime): {dup_count}")
    else:
        dup_count = df.duplicated().sum()
        print(f"{name:12s}  fully duplicated rows: {dup_count}")
```

### Remove Duplicates

```python
for name in ["telemetry", "errors", "maint", "failures"]:
    before = len(datasets[name])
    datasets[name] = datasets[name].drop_duplicates(
        subset=['machineID', 'datetime'], keep='first'
    ).reset_index(drop=True)
    after = len(datasets[name])
    if before != after:
        print(f"{name:12s}  removed {before - after} duplicates")
    else:
        print(f"{name:12s}  no duplicates ✓")

telemetry = datasets['telemetry']
errors    = datasets['errors']
maint     = datasets['maint']
failures  = datasets['failures']
```

---

## STAGE 8 — DATA CONSISTENCY CHECKS

### MachineID Validation

```python
all_machine_ids = set(machines['machineID'].unique())

for name in ["telemetry", "errors", "maint", "failures"]:
    df = datasets[name]
    ids = set(df['machineID'].unique())
    orphans = ids - all_machine_ids
    if orphans:
        print(f"⚠ {name:12s}  has machineIDs NOT in machines table: {orphans}")
    else:
        print(f"✓ {name:12s}  all machineIDs valid")
```

### Timestamp Range Validation

```python
telemetry_min = telemetry['datetime'].min()
telemetry_max = telemetry['datetime'].max()
print(f"Telemetry time range: {telemetry_min} → {telemetry_max}")

for name in ["errors", "maint", "failures"]:
    df = datasets[name]
    out_of_range = df[
        (df['datetime'] < telemetry_min) | (df['datetime'] > telemetry_max)
    ]
    if len(out_of_range) > 0:
        print(f"⚠ {name:12s}  {len(out_of_range)} events outside telemetry range")
    else:
        print(f"✓ {name:12s}  all timestamps within telemetry range")
```

### Domain Value Validation

```python
print("Error types:", errors['errorID'].unique())
print("Component types (maint):", maint['comp'].unique())
print("Failure types:", failures['failure'].unique())
```

### Failure–Maintenance Cross-Check

```python
for _, row in failures.iterrows():
    match = maint[
        (maint['machineID'] == row['machineID']) &
        (maint['comp'] == row['failure']) &
        (abs((maint['datetime'] - row['datetime']).dt.total_seconds()) <= 86400)
    ]
    if match.empty:
        print(f"⚠ Failure with no nearby maintenance: "
              f"Machine {row['machineID']}, {row['failure']}, {row['datetime']}")
```

---

## STAGE 9 — MERGING DATASETS

### Merge Machines (Static Metadata)

```python
master = telemetry.merge(machines, on='machineID', how='left')
print(f"After merging machines: {master.shape}")
```

### Merge Errors

```python
error_dummies = pd.get_dummies(errors, columns=['errorID'], prefix='', prefix_sep='')
error_dummies = error_dummies.groupby(['machineID', 'datetime']).sum().reset_index()

master = master.merge(error_dummies, on=['machineID', 'datetime'], how='left')

error_cols = [c for c in error_dummies.columns if c.startswith('error')]
master[error_cols] = master[error_cols].fillna(0).astype(int)

print(f"After merging errors: {master.shape}")
```

### Merge Maintenance

```python
maint_dummies = pd.get_dummies(maint, columns=['comp'], prefix='maint', prefix_sep='_')
maint_dummies = maint_dummies.groupby(['machineID', 'datetime']).sum().reset_index()

master = master.merge(maint_dummies, on=['machineID', 'datetime'], how='left')

maint_cols = [c for c in maint_dummies.columns if c.startswith('maint_')]
master[maint_cols] = master[maint_cols].fillna(0).astype(int)

print(f"After merging maintenance: {master.shape}")
```

### Merge Failures

```python
master = master.merge(failures, on=['machineID', 'datetime'], how='left')
master['failure'] = master['failure'].fillna('none')

print(f"After merging failures: {master.shape}")
print(f"\nFailure distribution:\n{master['failure'].value_counts()}")
```

### Verify Merge Integrity

```python
assert len(master) == len(telemetry), \
    f"Row count changed! Telemetry: {len(telemetry)}, Master: {len(master)}"

print(f"Final master shape: {master.shape}")
print(f"Columns: {list(master.columns)}")
print(master.head())
```

---

## STAGE 10 — FAILURE LABEL CREATION

### Binary Failure Label

```python
master['failed'] = (master['failure'] != 'none').astype(int)
print(f"Failure label distribution:\n{master['failed'].value_counts()}")
```

### Component-Specific Failure Labels

```python
failure_types = ['comp1', 'comp2', 'comp3', 'comp4']
for comp in failure_types:
    master[f'failure_{comp}'] = (master['failure'] == comp).astype(int)
```

### Predictive Label — "Will Fail in the Next 24 Hours?"

```python
PREDICTION_WINDOW = 24

def create_lookahead_label(group, window_hours=PREDICTION_WINDOW):
    label = pd.Series(0, index=group.index)
    failure_times = group[group['failed'] == 1]['datetime']

    for ft in failure_times:
        window_start = ft - pd.Timedelta(hours=window_hours)
        mask = (group['datetime'] >= window_start) & (group['datetime'] < ft)
        label[mask] = 1

    return label

master['failure_within_24h'] = (
    master.groupby('machineID', group_keys=False)
    .apply(create_lookahead_label)
)

print(f"\n'Failure within 24h' label distribution:")
print(master['failure_within_24h'].value_counts())
```

### Multi-Class Predictive Label (Optional)

```python
def create_component_lookahead(group, window_hours=PREDICTION_WINDOW):
    label = pd.Series('none', index=group.index)

    for comp in failure_types:
        failure_times = group[group['failure'] == comp]['datetime']
        for ft in failure_times:
            window_start = ft - pd.Timedelta(hours=window_hours)
            mask = (group['datetime'] >= window_start) & (group['datetime'] < ft)
            label[mask] = comp

    return label

master['component_at_risk'] = (
    master.groupby('machineID', group_keys=False)
    .apply(create_component_lookahead)
)

print(f"\nComponent-at-risk distribution:")
print(master['component_at_risk'].value_counts())
```

---

## STAGE 11 — CLASS IMBALANCE ANALYSIS

```python
print("=" * 50)
print("CLASS DISTRIBUTION — Binary Label")
print("=" * 50)
print(master['failed'].value_counts())
print(f"\nFailure rate: {master['failed'].mean() * 100:.4f}%")

print("\n" + "=" * 50)
print("CLASS DISTRIBUTION — 24h Lookahead Label")
print("=" * 50)
print(master['failure_within_24h'].value_counts())
print(f"\nPositive rate: {master['failure_within_24h'].mean() * 100:.2f}%")
```

### Class Weights

```python
from sklearn.utils.class_weight import compute_class_weight

weights = compute_class_weight(
    'balanced',
    classes=np.array([0, 1]),
    y=master['failure_within_24h']
)
class_weight_dict = {0: weights[0], 1: weights[1]}
print(f"Computed class weights: {class_weight_dict}")
```

### SMOTE (Apply Only to Training Set)

```python
# from imblearn.over_sampling import SMOTE
# smote = SMOTE(random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

### Random Oversampling

```python
# from imblearn.over_sampling import RandomOverSampler
# ros = RandomOverSampler(random_state=42)
# X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
```

### Random Undersampling

```python
# from imblearn.under_sampling import RandomUnderSampler
# rus = RandomUnderSampler(random_state=42)
# X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
```

---

## STAGE 12 — DATA NORMALIZATION / SCALING

### StandardScaler

```python
from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# Fit ONLY on training data; transform both train and test
# telemetry_scaled = scaler.fit_transform(telemetry[sensor_cols])
```

### MinMaxScaler

```python
from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler(feature_range=(0, 1))
# telemetry_scaled = scaler.fit_transform(telemetry[sensor_cols])
```

### RobustScaler

```python
from sklearn.preprocessing import RobustScaler

# scaler = RobustScaler()
# telemetry_scaled = scaler.fit_transform(telemetry[sensor_cols])
```

### Save Scaler

```python
# import joblib
# joblib.dump(scaler, 'sensor_scaler.pkl')
```

---

## STAGE 13 — FINAL PREPROCESSED DATASET VALIDATION

```python
print("=" * 60)
print("  FINAL DATASET VALIDATION")
print("=" * 60)

# 1. No missing values
missing = master.isnull().sum().sum()
print(f"\n1. Missing values: {missing}", "✓" if missing == 0 else "✗ INVESTIGATE!")

# 2. Sensor value ranges
for col in sensor_cols:
    lo, hi = master[col].min(), master[col].max()
    print(f"2. {col:12s}  range: [{lo:.2f}, {hi:.2f}]", end="  ")
    if col == 'volt' and (lo < 0 or hi > 300):
        print("⚠ Suspicious")
    elif col == 'pressure' and lo < 0:
        print("⚠ Negative pressure!")
    else:
        print("✓")

# 3. Time ordering
time_sorted = master.groupby('machineID')['datetime'].apply(
    lambda x: x.is_monotonic_increasing
).all()
print(f"\n3. Time ordering correct: {time_sorted}", "✓" if time_sorted else "✗ RE-SORT!")

# 4. Failure labels exist
print(f"\n4. Failure label columns present:")
for col in ['failed', 'failure_within_24h']:
    if col in master.columns:
        print(f"   {col}: ✓  (unique values: {master[col].unique()})")
    else:
        print(f"   {col}: ✗ MISSING!")

# 5. Shape & class distribution
print(f"\n5. Final dataset shape: {master.shape}")
print(f"   Rows: {len(master):,}")
print(f"   Columns: {master.shape[1]}")
print(f"\n   24h failure label distribution:")
print(f"   {master['failure_within_24h'].value_counts().to_dict()}")
ratio = master['failure_within_24h'].value_counts()
print(f"   Imbalance ratio: 1:{ratio[0]//max(ratio[1],1)}")

# 6. Column overview
print(f"\n6. All columns and types:")
print(master.dtypes)
```

### Save Clean Dataset

```python
output_path = r"d:\HackMasters\Dataset\master_preprocessed.csv"
master.to_csv(output_path, index=False)
print(f"\n✅ Preprocessed dataset saved to: {output_path}")
print(f"   Shape: {master.shape}")
```
