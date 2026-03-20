"""
Feature Engineering Pipeline - Predictive Maintenance
=====================================================
Input  : master_preprocessed.csv (from preprocessing pipeline)
Output : master_featured.csv
Steps  : 15 feature engineering stages + final validation
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings
import os
import sys

warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "plots")

INPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "master_preprocessed.csv")
OUTPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "master_featured.csv")

SENSOR_COLS = ["volt", "rotate", "pressure", "vibration"]
ERROR_COLS = ["error1", "error2", "error3", "error4", "error5"]
MAINT_COLS = ["maint_comp1", "maint_comp2", "maint_comp3", "maint_comp4"]
LAGS = [1, 3, 6, 12, 24]
WINDOWS = [3, 6, 12, 24]


# ---------------------------------------------------------------
def time_since_event(series):
    """Count hours since the last 1 in a binary series."""
    result = np.full(len(series), np.nan)
    last = np.nan
    vals = series.values
    for i in range(len(vals)):
        if vals[i] > 0:
            last = 0.0
        elif not np.isnan(last):
            last += 1.0
        result[i] = last
    return pd.Series(result, index=series.index)


# ===============================================================
#  STEP 1 -- Lag Features (past sensor values)
# ===============================================================
def step_01_lags(df):
    print("\n" + "=" * 60)
    print("  STEP 1 -- Lag Features")
    print("=" * 60)

    for col in SENSOR_COLS:
        for lag in LAGS:
            df[f"{col}_lag_{lag}"] = df.groupby("machineID")[col].shift(lag)

    created = [c for c in df.columns if "_lag_" in c]
    print(f"  Created {len(created)} lag features")
    return df


# ===============================================================
#  STEP 2 -- Rolling Window Statistics
# ===============================================================
def step_02_rolling(df):
    print("\n" + "=" * 60)
    print("  STEP 2 -- Rolling Window Statistics")
    print("=" * 60)

    count = 0
    for col in SENSOR_COLS:
        for w in WINDOWS:
            grp = df.groupby("machineID")[col]
            df[f"{col}_rmean_{w}"] = grp.transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
            df[f"{col}_rstd_{w}"] = grp.transform(
                lambda x: x.rolling(w, min_periods=1).std()
            )
            df[f"{col}_rmin_{w}"] = grp.transform(
                lambda x: x.rolling(w, min_periods=1).min()
            )
            df[f"{col}_rmax_{w}"] = grp.transform(
                lambda x: x.rolling(w, min_periods=1).max()
            )
            count += 4

    print(f"  Created {count} rolling features")
    return df


# ===============================================================
#  STEP 3 -- Sensor Trend Features
# ===============================================================
def step_03_trends(df):
    print("\n" + "=" * 60)
    print("  STEP 3 -- Sensor Trend Features")
    print("=" * 60)

    for col in SENSOR_COLS:
        short = df.groupby("machineID")[col].transform(
            lambda x: x.rolling(6, min_periods=1).mean()
        )
        long = df.groupby("machineID")[col].transform(
            lambda x: x.rolling(24, min_periods=1).mean()
        )
        df[f"{col}_trend"] = short - long

    print(f"  Created {len(SENSOR_COLS)} trend features")
    return df


# ===============================================================
#  STEP 4 -- Rate of Change Features
# ===============================================================
def step_04_rate_of_change(df):
    print("\n" + "=" * 60)
    print("  STEP 4 -- Rate of Change Features")
    print("=" * 60)

    for col in SENSOR_COLS:
        df[f"{col}_change"] = df.groupby("machineID")[col].diff()

    print(f"  Created {len(SENSOR_COLS)} rate-of-change features")
    return df


# ===============================================================
#  STEP 5 -- Duration Above Threshold Features
# ===============================================================
def step_05_duration_above(df):
    print("\n" + "=" * 60)
    print("  STEP 5 -- Duration Above Threshold")
    print("=" * 60)

    for col in SENSOR_COLS:
        threshold = df[col].quantile(0.95)
        above = (df[col] > threshold).astype(int)

        def cumcount_runs(x):
            groups = (x != x.shift()).cumsum()
            return x.groupby(groups).cumcount() + 1

        df[f"{col}_high_duration"] = (
            above.groupby(df["machineID"]).transform(cumcount_runs) * above
        )
        print(f"    {col:12s}  threshold={threshold:.2f}")

    print(f"  Created {len(SENSOR_COLS)} duration features")
    return df


# ===============================================================
#  STEP 6 -- Error Frequency Features
# ===============================================================
def step_06_errors(df):
    print("\n" + "=" * 60)
    print("  STEP 6 -- Error Frequency Features")
    print("=" * 60)

    df["total_errors"] = df[ERROR_COLS].sum(axis=1)

    for w in [24, 72]:
        df[f"error_count_{w}h"] = df.groupby("machineID")["total_errors"].transform(
            lambda x: x.rolling(w, min_periods=1).sum()
        )
        print(f"    error_count_{w}h created")

    # Per error type rolling counts
    for ecol in ERROR_COLS:
        df[f"{ecol}_count_24h"] = df.groupby("machineID")[ecol].transform(
            lambda x: x.rolling(24, min_periods=1).sum()
        )

    print(f"  Created {2 + len(ERROR_COLS)} error frequency features")
    return df


# ===============================================================
#  STEP 7 -- Maintenance History Features
# ===============================================================
def step_07_maintenance(df):
    print("\n" + "=" * 60)
    print("  STEP 7 -- Maintenance History Features")
    print("=" * 60)

    df["any_maint"] = df[MAINT_COLS].max(axis=1)

    # Hours since last maintenance
    df["hours_since_maint"] = df.groupby("machineID")["any_maint"].transform(
        time_since_event
    )
    print("    hours_since_maint created")

    # Maintenance count in last 30 days (720 hours)
    df["maint_count_30d"] = df.groupby("machineID")["any_maint"].transform(
        lambda x: x.rolling(720, min_periods=1).sum()
    )
    print("    maint_count_30d created")

    # Per-component hours since last maintenance
    for mc in MAINT_COLS:
        comp_name = mc.replace("maint_", "")
        df[f"hours_since_{comp_name}_maint"] = df.groupby("machineID")[mc].transform(
            time_since_event
        )
    print(f"    Per-component time-since-maint created ({len(MAINT_COLS)} cols)")

    return df


# ===============================================================
#  STEP 8 -- Machine Age Features
# ===============================================================
def step_08_age(df):
    print("\n" + "=" * 60)
    print("  STEP 8 -- Machine Age Features")
    print("=" * 60)

    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 5, 10, 15, 20, 100],
        labels=["new", "young", "mid", "old", "very_old"],
    )
    print(f"  Age group distribution:\n{df['age_group'].value_counts().to_string()}")
    return df


# ===============================================================
#  STEP 9 -- Failure History Features
# ===============================================================
def step_09_failure_history(df):
    print("\n" + "=" * 60)
    print("  STEP 9 -- Failure History Features")
    print("=" * 60)

    df["cumulative_failures"] = df.groupby("machineID")["failed"].cumsum()

    df["hours_since_failure"] = df.groupby("machineID")["failed"].transform(
        time_since_event
    )
    print(f"  cumulative_failures max: {df['cumulative_failures'].max()}")
    print(f"  hours_since_failure non-null: {df['hours_since_failure'].notna().sum():,}")
    return df


# ===============================================================
#  STEP 10 -- Machine Health Score
# ===============================================================
def step_10_health_score(df):
    print("\n" + "=" * 60)
    print("  STEP 10 -- Machine Health Score")
    print("=" * 60)

    scaler = MinMaxScaler()
    normed = pd.DataFrame(
        scaler.fit_transform(df[SENSOR_COLS]),
        columns=SENSOR_COLS,
        index=df.index,
    )

    weights = {"volt": 0.20, "rotate": 0.30, "pressure": 0.25, "vibration": 0.25}
    df["health_score"] = sum(normed[c] * w for c, w in weights.items())

    print(f"  health_score stats:\n{df['health_score'].describe().to_string()}")
    return df


# ===============================================================
#  STEP 11 -- Operating Condition Interaction Features
# ===============================================================
def step_11_interactions(df):
    print("\n" + "=" * 60)
    print("  STEP 11 -- Interaction Features")
    print("=" * 60)

    df["pressure_rotate_ratio"] = df["pressure"] / df["rotate"].replace(0, np.nan)
    df["vibration_rotate_ratio"] = df["vibration"] / df["rotate"].replace(0, np.nan)
    df["volt_pressure_product"] = df["volt"] * df["pressure"]

    print("  Created 3 interaction features")
    return df


# ===============================================================
#  STEP 12 -- Error Type Encoding (verify existing)
# ===============================================================
def step_12_error_encoding(df):
    print("\n" + "=" * 60)
    print("  STEP 12 -- Error Type Encoding (verify)")
    print("=" * 60)

    print("  Error column sums:")
    for c in ERROR_COLS:
        print(f"    {c}: {df[c].sum():,}")
    return df


# ===============================================================
#  STEP 13 -- Component Failure Encoding (verify existing)
# ===============================================================
def step_13_failure_encoding(df):
    print("\n" + "=" * 60)
    print("  STEP 13 -- Component Failure Encoding (verify)")
    print("=" * 60)

    fail_cols = ["failure_comp1", "failure_comp2", "failure_comp3", "failure_comp4"]
    for c in fail_cols:
        print(f"    {c}: {df[c].sum()}")
    return df


# ===============================================================
#  STEP 14 -- Time-Based Features
# ===============================================================
def step_14_time_features(df):
    print("\n" + "=" * 60)
    print("  STEP 14 -- Time-Based Features")
    print("=" * 60)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour_of_day"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month

    print("  Created: hour_of_day, day_of_week, month")
    return df


# ===============================================================
#  STEP 15 -- Sensor Stability Features (rolling variance)
# ===============================================================
def step_15_stability(df):
    print("\n" + "=" * 60)
    print("  STEP 15 -- Sensor Stability Features")
    print("=" * 60)

    count = 0
    for col in SENSOR_COLS:
        for w in [12, 24]:
            df[f"{col}_var_{w}h"] = df.groupby("machineID")[col].transform(
                lambda x: x.rolling(w, min_periods=1).var()
            )
            count += 1

    print(f"  Created {count} variance features")
    return df


# ===============================================================
#  FINAL STEP -- Feature Validation
# ===============================================================
def step_final_validate(df):
    print("\n" + "=" * 60)
    print("  FINAL STEP -- Feature Validation")
    print("=" * 60)

    # Fill any remaining NaN from lag/diff/rolling edges
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    nan_before = df[numeric_cols].isnull().sum().sum()
    print(f"\n  NaN before fill: {nan_before:,}")

    df[numeric_cols] = df[numeric_cols].fillna(0)

    nan_after = df[numeric_cols].isnull().sum().sum()
    print(f"  NaN after fill : {nan_after}")

    # Shape
    print(f"\n  Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Top features correlated with failure label
    print("\n  Top 25 features correlated with failure_within_24h:")
    corr = df[numeric_cols].corrwith(df["failure_within_24h"]).abs().sort_values(ascending=False)
    for feat, val in corr.head(25).items():
        print(f"    {feat:40s}  r = {val:.4f}")

    # Correlation heatmap (top features only)
    top_feats = list(corr.head(15).index)
    if "failure_within_24h" not in top_feats:
        top_feats.append("failure_within_24h")
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(df[top_feats].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax, square=True, linewidths=0.5)
    ax.set_title("Top Features - Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "feature_correlation_heatmap.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("  [OK] Correlation heatmap saved")

    # Class distribution
    print(f"\n  Target distribution (failure_within_24h):")
    print(df["failure_within_24h"].value_counts().to_string())

    # Column listing
    print(f"\n  All {df.shape[1]} columns:")
    for i, col in enumerate(df.columns):
        print(f"    {i+1:3d}. {col:40s} {df[col].dtype}")

    return df


# ===============================================================
#  MAIN
# ===============================================================
def main():
    print("\n" + "#" * 60)
    print("  FEATURE ENGINEERING PIPELINE")
    print("#" * 60)

    # Load preprocessed data
    print("\n  Loading preprocessed dataset...")
    df = pd.read_csv(INPUT_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    print(f"  Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

    df = step_01_lags(df)
    df = step_02_rolling(df)
    df = step_03_trends(df)
    df = step_04_rate_of_change(df)
    df = step_05_duration_above(df)
    df = step_06_errors(df)
    df = step_07_maintenance(df)
    df = step_08_age(df)
    df = step_09_failure_history(df)
    df = step_10_health_score(df)
    df = step_11_interactions(df)
    df = step_12_error_encoding(df)
    df = step_13_failure_encoding(df)
    df = step_14_time_features(df)
    df = step_15_stability(df)
    df = step_final_validate(df)

    # Save
    print(f"\n  Saving to {OUTPUT_PATH}...")
    df.to_csv(OUTPUT_PATH, index=False)
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"  [DONE] Saved: {size_mb:.1f} MB, {df.shape}")

    print("\n" + "#" * 60)
    print("  FEATURE ENGINEERING COMPLETE")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
