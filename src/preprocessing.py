"""
Predictive Maintenance - Complete Data Cleaning and Preprocessing Pipeline
==========================================================================
Dataset : Microsoft Azure Predictive Maintenance (PdM)
Stages  : 13 (Dataset Understanding -> Final Validation and Export)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
import sys

warnings.filterwarnings("ignore")

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "plots")

OUTPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "master_preprocessed.csv")

SENSOR_COLS = ["volt", "rotate", "pressure", "vibration"]
PREDICTION_WINDOW = 24  # hours
ROLLING_WINDOW = 24     # hours
EWM_SPAN = 24           # hours


# ===============================================================
#  STAGE 1 -- DATASET UNDERSTANDING
# ===============================================================
def stage_01_understand(telemetry, errors, maint, failures, machines):
    print("\n" + "=" * 70)
    print("  STAGE 1 -- DATASET UNDERSTANDING")
    print("=" * 70)

    datasets = {
        "telemetry": telemetry,
        "errors": errors,
        "maint": maint,
        "failures": failures,
        "machines": machines,
    }

    for name, df in datasets.items():
        print(f"\n{'-' * 50}")
        print(f"  {name.upper()}")
        print(f"{'-' * 50}")
        print(f"  Shape   : {df.shape}")
        print(f"  Columns : {list(df.columns)}")
        print(f"\n  Data types:")
        for col in df.columns:
            print(f"    {col:20s} -> {df[col].dtype}")
        print(f"\n  First 3 rows:")
        print(df.head(3).to_string(index=False))

    # Primary & join keys
    print(f"\n{'-' * 50}")
    print("  JOIN KEY ANALYSIS")
    print(f"{'-' * 50}")
    print(f"  Unique machineIDs in machines: {machines['machineID'].nunique()}")
    for name, df in datasets.items():
        if "machineID" in df.columns:
            print(
                f"  {name:12s} -> machineID range: "
                f"{df['machineID'].min()} - {df['machineID'].max()}, "
                f"unique: {df['machineID'].nunique()}"
            )

    # Time coverage
    print(f"\n{'-' * 50}")
    print("  TIME COVERAGE")
    print(f"{'-' * 50}")
    for name in ["telemetry", "errors", "maint", "failures"]:
        df = datasets[name]
        print(f"  {name:12s} -> {df['datetime'].min()}  to  {df['datetime'].max()}")

    # Machine metadata
    print(f"\n{'-' * 50}")
    print("  MACHINE METADATA")
    print(f"{'-' * 50}")
    print(f"\n  Model distribution:")
    print(machines["model"].value_counts().to_string())
    print(f"\n  Age statistics:")
    print(machines["age"].describe().to_string())

    # Sensor distributions
    print(f"\n{'-' * 50}")
    print("  SENSOR STATISTICS")
    print(f"{'-' * 50}")
    print(telemetry[SENSOR_COLS].describe().to_string())

    # Histograms
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, col in zip(axes.flatten(), SENSOR_COLS):
        ax.hist(telemetry[col].dropna(), bins=60, edgecolor="black", alpha=0.7)
        ax.set_title(col, fontsize=13)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
    plt.suptitle("Stage 1 - Raw Sensor Distributions", fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "stage1_distributions.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("\n  [OK] Histogram saved -> stage1_distributions.png")


# ===============================================================
#  STAGE 2 -- DATA TYPE CORRECTION
# ===============================================================
def stage_02_types(telemetry, errors, maint, failures, machines):
    print("\n" + "=" * 70)
    print("  STAGE 2 -- DATA TYPE CORRECTION")
    print("=" * 70)

    # Datetime conversion
    for name, df in [("telemetry", telemetry), ("errors", errors),
                     ("maint", maint), ("failures", failures)]:
        df["datetime"] = pd.to_datetime(df["datetime"])
        print(f"  {name:12s}  datetime dtype -> {df['datetime'].dtype}")

    # Chronological-order check
    for name, df in [("telemetry", telemetry), ("errors", errors),
                     ("maint", maint), ("failures", failures)]:
        diffs = df.groupby("machineID")["datetime"].diff()
        backward = (diffs < pd.Timedelta(0)).sum()
        print(f"  {name:12s}  backward jumps  -> {backward}")

    # Numeric conversion
    for col in SENSOR_COLS:
        telemetry[col] = pd.to_numeric(telemetry[col], errors="coerce")
    print(f"\n  Sensor dtypes: {telemetry[SENSOR_COLS].dtypes.to_dict()}")

    # Categorical conversion
    machines["model"] = machines["model"].astype("category")
    errors["errorID"] = errors["errorID"].astype("category")
    maint["comp"] = maint["comp"].astype("category")
    # Keep failures['failure'] as object/string to allow fillna('none') after merge

    print("  [OK] Categorical conversions applied")
    return telemetry, errors, maint, failures, machines


# ===============================================================
#  STAGE 3 -- TIME SERIES SORTING
# ===============================================================
def stage_03_sort(telemetry, errors, maint, failures):
    print("\n" + "=" * 70)
    print("  STAGE 3 -- TIME SERIES SORTING")
    print("=" * 70)

    telemetry = telemetry.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    errors = errors.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    maint = maint.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    failures = failures.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    print("  [OK] All tables sorted by (machineID, datetime)")

    # Verify uniform hourly intervals in telemetry
    time_diffs = telemetry.groupby("machineID")["datetime"].diff()
    non_hourly = time_diffs[time_diffs != pd.Timedelta(hours=1)]
    n_gaps = non_hourly.dropna().shape[0]
    print(f"  Non-hourly gaps in telemetry: {n_gaps}")
    if n_gaps > 0:
        print("  Sample of irregular gaps:")
        print(non_hourly.dropna().value_counts().head(5).to_string())

    return telemetry, errors, maint, failures


# ===============================================================
#  STAGE 4 -- MISSING VALUE ANALYSIS & IMPUTATION
# ===============================================================
def stage_04_missing(telemetry, errors, maint, failures, machines):
    print("\n" + "=" * 70)
    print("  STAGE 4 -- MISSING VALUE ANALYSIS")
    print("=" * 70)

    datasets = {
        "telemetry": telemetry,
        "errors": errors,
        "maint": maint,
        "failures": failures,
        "machines": machines,
    }

    for name, df in datasets.items():
        total = df.isnull().sum()
        pct = (df.isnull().sum() / len(df)) * 100
        missing_df = pd.DataFrame({"Missing": total, "Percent": pct})
        missing_df = missing_df[missing_df["Missing"] > 0]
        if missing_df.empty:
            print(f"  {name:12s} -> No missing values [OK]")
        else:
            print(f"\n  {name:12s} -> MISSING VALUES DETECTED:")
            print(missing_df.to_string())

    # Pattern analysis (per-machine)
    has_missing = telemetry[SENSOR_COLS].isnull().any().any()
    if has_missing:
        machine_missing = telemetry.groupby("machineID")[SENSOR_COLS].apply(
            lambda x: x.isnull().sum()
        )
        nonzero = machine_missing[machine_missing.sum(axis=1) > 0]
        if len(nonzero) > 0:
            print(f"\n  Machines with missing sensor values:")
            print(nonzero.to_string())

    # Imputation: linear interpolation -> ffill -> bfill
    print("\n  Applying imputation: interpolation -> ffill -> bfill ...")

    for col in SENSOR_COLS:
        telemetry[col] = telemetry.groupby("machineID")[col].transform(
            lambda x: x.interpolate(method="linear")
        )

    for col in SENSOR_COLS:
        telemetry[col] = telemetry.groupby("machineID")[col].transform(
            lambda x: x.ffill()
        )

    for col in SENSOR_COLS:
        telemetry[col] = telemetry.groupby("machineID")[col].transform(
            lambda x: x.bfill()
        )

    remaining = telemetry[SENSOR_COLS].isnull().sum().sum()
    assert remaining == 0, f"Still have {remaining} NaNs after imputation!"
    print("  [OK] All missing sensor values imputed (0 NaN remaining)")

    return telemetry


# ===============================================================
#  STAGE 5 -- OUTLIER DETECTION & CAPPING
# ===============================================================
def stage_05_outliers(telemetry):
    print("\n" + "=" * 70)
    print("  STAGE 5 -- OUTLIER DETECTION")
    print("=" * 70)

    # Z-Score analysis
    print("\n  Z-Score analysis (|z| > 3):")
    for col in SENSOR_COLS:
        z = np.abs(stats.zscore(telemetry[col].dropna()))
        n = (z > 3).sum()
        print(f"    {col:12s}  outliers: {n:>6,} ({n / len(telemetry) * 100:.3f}%)")

    # IQR analysis
    print("\n  IQR analysis (1.5 x IQR):")
    for col in SENSOR_COLS:
        Q1 = telemetry[col].quantile(0.25)
        Q3 = telemetry[col].quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n = ((telemetry[col] < lo) | (telemetry[col] > hi)).sum()
        print(f"    {col:12s}  outliers: {n:>6,}  (valid range: {lo:.2f} - {hi:.2f})")

    # Domain-specific checks
    print("\n  Domain-specific checks:")
    print(f"    Voltage = 0       : {(telemetry['volt'] == 0).sum()}")
    print(f"    Negative pressure : {(telemetry['pressure'] < 0).sum()}")
    print(f"    Vibration > 100   : {(telemetry['vibration'] > 100).sum()}")
    print(f"    Rotation > 600    : {(telemetry['rotate'] > 600).sum()}")

    # Visualizations -- Box plots
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for i, col in enumerate(SENSOR_COLS):
        telemetry.boxplot(column=col, ax=axes[i])
        axes[i].set_title(col, fontsize=12)
    plt.suptitle("Stage 5 - Box Plots (Before Capping)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "stage5_boxplots_before.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, col in zip(axes.flatten(), SENSOR_COLS):
        sns.histplot(telemetry[col], bins=100, kde=True, ax=ax)
        ax.set_title(f"{col} Distribution", fontsize=12)
    plt.suptitle("Stage 5 - Distributions (Before Capping)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "stage5_distributions_before.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # Time-series sample
    sample_machine = 1
    machine_data = telemetry[telemetry["machineID"] == sample_machine]
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    for ax, col in zip(axes, SENSOR_COLS):
        ax.plot(machine_data["datetime"], machine_data[col], linewidth=0.5)
        ax.set_ylabel(col, fontsize=11)
    axes[0].set_title(f"Machine {sample_machine} - Sensor Time Series", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "stage5_timeseries_machine1.png"), dpi=120, bbox_inches="tight")
    plt.close()

    print("  [OK] Visualizations saved (boxplots, distributions, time-series)")

    # Winsorize at 1st / 99th percentile
    print("\n  Winsorizing at 1st / 99th percentile:")
    for col in SENSOR_COLS:
        lo = telemetry[col].quantile(0.01)
        hi = telemetry[col].quantile(0.99)
        telemetry[col] = telemetry[col].clip(lower=lo, upper=hi)
        print(f"    {col:12s}  clipped to [{lo:.2f}, {hi:.2f}]")

    print("  [OK] Outlier capping complete")
    return telemetry


# ===============================================================
#  STAGE 6 -- SENSOR DATA SMOOTHING
# ===============================================================
def stage_06_smoothing(telemetry):
    print("\n" + "=" * 70)
    print("  STAGE 6 -- SENSOR DATA SMOOTHING")
    print("=" * 70)

    # Rolling mean
    for col in SENSOR_COLS:
        telemetry[f"{col}_rolling_mean"] = telemetry.groupby("machineID")[col].transform(
            lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        )
    print(f"  [OK] Rolling mean ({ROLLING_WINDOW}h) columns added")

    # Exponential smoothing
    for col in SENSOR_COLS:
        telemetry[f"{col}_ewm"] = telemetry.groupby("machineID")[col].transform(
            lambda x: x.ewm(span=EWM_SPAN, min_periods=1).mean()
        )
    print(f"  [OK] EWM (span={EWM_SPAN}) columns added")

    # Visualization
    sample = telemetry[telemetry["machineID"] == 1].head(500)
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    for ax, col in zip(axes, SENSOR_COLS):
        ax.plot(sample["datetime"], sample[col], alpha=0.3, label="Raw")
        ax.plot(sample["datetime"], sample[f"{col}_rolling_mean"], label=f"Rolling Mean ({ROLLING_WINDOW}h)")
        ax.plot(sample["datetime"], sample[f"{col}_ewm"], label=f"EWM (span={EWM_SPAN})")
        ax.set_ylabel(col, fontsize=11)
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_title("Machine 1 - Raw vs Smoothed Signals", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "stage6_smoothing_comparison.png"), dpi=120, bbox_inches="tight")
    plt.close()

    print("  [OK] Smoothing comparison plot saved")
    print(f"  Telemetry columns now: {list(telemetry.columns)}")
    return telemetry


# ===============================================================
#  STAGE 7 -- DUPLICATE DATA HANDLING
# ===============================================================
def stage_07_duplicates(telemetry, errors, maint, failures, machines):
    print("\n" + "=" * 70)
    print("  STAGE 7 -- DUPLICATE DATA HANDLING")
    print("=" * 70)

    result = {}
    for name, df in [("telemetry", telemetry), ("errors", errors),
                     ("maint", maint), ("failures", failures)]:
        before = len(df)
        df = df.drop_duplicates(subset=["machineID", "datetime"], keep="first").reset_index(drop=True)
        removed = before - len(df)
        result[name] = df
        if removed:
            print(f"  {name:12s}  removed {removed} duplicates")
        else:
            print(f"  {name:12s}  no duplicates [OK]")

    # Machines table -- no datetime, check full row dups
    before = len(machines)
    machines = machines.drop_duplicates().reset_index(drop=True)
    removed = before - len(machines)
    if removed:
        print(f"  machines      removed {removed} duplicates")
    else:
        print(f"  machines      no duplicates [OK]")

    return result["telemetry"], result["errors"], result["maint"], result["failures"], machines


# ===============================================================
#  STAGE 8 -- DATA CONSISTENCY CHECKS
# ===============================================================
def stage_08_consistency(telemetry, errors, maint, failures, machines):
    print("\n" + "=" * 70)
    print("  STAGE 8 -- DATA CONSISTENCY CHECKS")
    print("=" * 70)

    all_ids = set(machines["machineID"].unique())

    # machineID validation
    for name, df in [("telemetry", telemetry), ("errors", errors),
                     ("maint", maint), ("failures", failures)]:
        ids = set(df["machineID"].unique())
        orphans = ids - all_ids
        if orphans:
            print(f"  [WARN] {name:12s}  orphan machineIDs: {orphans}")
        else:
            print(f"  [OK]   {name:12s}  all machineIDs valid")

    # Timestamp range validation
    t_min = telemetry["datetime"].min()
    t_max = telemetry["datetime"].max()
    print(f"\n  Telemetry range: {t_min} -> {t_max}")
    for name, df in [("errors", errors), ("maint", maint), ("failures", failures)]:
        out = df[(df["datetime"] < t_min) | (df["datetime"] > t_max)]
        if len(out) > 0:
            print(f"  [WARN] {name:12s}  {len(out)} events outside telemetry range")
        else:
            print(f"  [OK]   {name:12s}  all timestamps within range")

    # Domain value validation
    print(f"\n  Error types   : {sorted(errors['errorID'].unique())}")
    print(f"  Component types: {sorted(maint['comp'].unique())}")
    print(f"  Failure types  : {sorted(failures['failure'].unique())}")

    # Failure-Maintenance cross-check
    print(f"\n  Failure <-> Maintenance cross-check:")
    unmatched = 0
    for _, row in failures.iterrows():
        match = maint[
            (maint["machineID"] == row["machineID"])
            & (maint["comp"] == row["failure"])
            & (abs((maint["datetime"] - row["datetime"]).dt.total_seconds()) <= 86400)
        ]
        if match.empty:
            unmatched += 1
    if unmatched == 0:
        print("  [OK] Every failure has a matching maintenance record within 24h")
    else:
        print(f"  [WARN] {unmatched} failure(s) with no matching maintenance within 24h")


# ===============================================================
#  STAGE 9 -- MERGING DATASETS
# ===============================================================
def stage_09_merge(telemetry, errors, maint, failures, machines):
    print("\n" + "=" * 70)
    print("  STAGE 9 -- MERGING DATASETS")
    print("=" * 70)

    original_len = len(telemetry)

    # 1. Merge machines metadata
    master = telemetry.merge(machines, on="machineID", how="left")
    print(f"  After merging machines     : {master.shape}")

    # 2. Merge errors (one-hot encoded)
    error_dummies = pd.get_dummies(errors, columns=["errorID"], prefix="", prefix_sep="")
    error_dummies = error_dummies.groupby(["machineID", "datetime"]).sum().reset_index()
    master = master.merge(error_dummies, on=["machineID", "datetime"], how="left")
    error_cols = [c for c in error_dummies.columns if c.startswith("error")]
    master[error_cols] = master[error_cols].fillna(0).astype(int)
    print(f"  After merging errors       : {master.shape}")

    # 3. Merge maintenance (one-hot encoded)
    maint_dummies = pd.get_dummies(maint, columns=["comp"], prefix="maint", prefix_sep="_")
    maint_dummies = maint_dummies.groupby(["machineID", "datetime"]).sum().reset_index()
    master = master.merge(maint_dummies, on=["machineID", "datetime"], how="left")
    maint_cols = [c for c in maint_dummies.columns if c.startswith("maint_")]
    master[maint_cols] = master[maint_cols].fillna(0).astype(int)
    print(f"  After merging maintenance  : {master.shape}")

    # 4. Merge failures (ensure failure col is string for fillna)
    failures_str = failures.copy()
    failures_str["failure"] = failures_str["failure"].astype(str)
    master = master.merge(failures_str, on=["machineID", "datetime"], how="left")
    master["failure"] = master["failure"].fillna("none")
    print(f"  After merging failures     : {master.shape}")
    print(f"\n  Failure distribution:\n{master['failure'].value_counts().to_string()}")

    # 5. Verify integrity
    assert len(master) == original_len, (
        f"Row count changed! Telemetry: {original_len}, Master: {len(master)}"
    )
    print(f"\n  [OK] Merge integrity verified (rows = {len(master):,})")

    return master


# ===============================================================
#  STAGE 10 -- FAILURE LABEL CREATION
# ===============================================================
def stage_10_labels(master):
    print("\n" + "=" * 70)
    print("  STAGE 10 -- FAILURE LABEL CREATION")
    print("=" * 70)

    failure_types = ["comp1", "comp2", "comp3", "comp4"]

    # Binary failure label
    master["failed"] = (master["failure"] != "none").astype(int)
    print(f"  Binary failure label:\n{master['failed'].value_counts().to_string()}")

    # Per-component labels
    for comp in failure_types:
        master[f"failure_{comp}"] = (master["failure"] == comp).astype(int)
    print("  [OK] Component-specific failure columns added")

    # 24-hour lookahead label
    print(f"\n  Creating {PREDICTION_WINDOW}h lookahead label (this may take a moment)...")

    labels_24h = pd.Series(0, index=master.index, dtype=int)
    for machine_id, group in master.groupby("machineID"):
        failure_times = group.loc[group["failed"] == 1, "datetime"]
        for ft in failure_times:
            window_start = ft - pd.Timedelta(hours=PREDICTION_WINDOW)
            mask = (group["datetime"] >= window_start) & (group["datetime"] < ft)
            labels_24h[group.index[mask.values]] = 1

    master["failure_within_24h"] = labels_24h
    print(f"  Failure-within-24h distribution:\n{master['failure_within_24h'].value_counts().to_string()}")

    # Multi-class lookahead
    print(f"\n  Creating component-at-risk label...")
    labels_comp = pd.Series("none", index=master.index)
    for machine_id, group in master.groupby("machineID"):
        for comp in failure_types:
            failure_times = group.loc[group["failure"] == comp, "datetime"]
            for ft in failure_times:
                window_start = ft - pd.Timedelta(hours=PREDICTION_WINDOW)
                mask = (group["datetime"] >= window_start) & (group["datetime"] < ft)
                labels_comp[group.index[mask.values]] = comp

    master["component_at_risk"] = labels_comp
    print(f"  Component-at-risk distribution:\n{master['component_at_risk'].value_counts().to_string()}")

    return master


# ===============================================================
#  STAGE 11 -- CLASS IMBALANCE ANALYSIS
# ===============================================================
def stage_11_imbalance(master):
    print("\n" + "=" * 70)
    print("  STAGE 11 -- CLASS IMBALANCE ANALYSIS")
    print("=" * 70)

    print("\n  Binary label ('failed'):")
    print(master["failed"].value_counts().to_string())
    print(f"  Failure rate: {master['failed'].mean() * 100:.4f}%")

    print(f"\n  24h lookahead label ('failure_within_24h'):")
    print(master["failure_within_24h"].value_counts().to_string())
    print(f"  Positive rate: {master['failure_within_24h'].mean() * 100:.2f}%")

    ratio = master["failure_within_24h"].value_counts()
    denom = max(ratio.get(1, 1), 1)
    print(f"  Imbalance ratio (normal : failure) ~ {ratio[0] // denom} : 1")

    # Compute class weights
    from sklearn.utils.class_weight import compute_class_weight

    weights = compute_class_weight(
        "balanced",
        classes=np.array([0, 1]),
        y=master["failure_within_24h"],
    )
    class_weight_dict = {0: weights[0], 1: weights[1]}
    print(f"\n  Computed class weights: {class_weight_dict}")
    print("  (Use these in your classifier: class_weight=... or scale_pos_weight=...)")

    # Visual
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    master["failed"].value_counts().plot.bar(ax=axes[0], color=["#2ecc71", "#e74c3c"])
    axes[0].set_title("Binary Failure Label")
    axes[0].set_xticklabels(["Normal (0)", "Failure (1)"], rotation=0)

    master["failure_within_24h"].value_counts().plot.bar(ax=axes[1], color=["#2ecc71", "#e74c3c"])
    axes[1].set_title("Failure Within 24h Label")
    axes[1].set_xticklabels(["Normal (0)", "At Risk (1)"], rotation=0)

    plt.suptitle("Stage 11 - Class Distribution", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "stage11_class_distribution.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("  [OK] Class distribution plot saved")


# ===============================================================
#  STAGE 12 -- DATA NORMALIZATION / SCALING (info only)
# ===============================================================
def stage_12_scaling_info():
    print("\n" + "=" * 70)
    print("  STAGE 12 -- NORMALIZATION / SCALING (INFO)")
    print("=" * 70)
    print("""
  Scaling should be applied AFTER train/test split to prevent data leakage.
  
  Recommended scalers:
    - StandardScaler : for normally distributed features
    - MinMaxScaler   : for bounded-range requirements (neural nets)
    - RobustScaler   : when outliers remain in data
  
  Tree-based models (RF, XGBoost, LightGBM) do NOT require scaling.
  
  Workflow:
    1. Split data -> train / test
    2. Fit scaler on training data ONLY
    3. Transform both train and test with the fitted scaler
    4. Save scaler: joblib.dump(scaler, 'sensor_scaler.pkl')
  
  [OK] No transformation applied at this stage (to be done during modelling).
""")


# ===============================================================
#  STAGE 13 -- FINAL VALIDATION & EXPORT
# ===============================================================
def stage_13_validate(master):
    print("\n" + "=" * 70)
    print("  STAGE 13 -- FINAL DATASET VALIDATION")
    print("=" * 70)

    # 1. Missing values
    missing = master.isnull().sum().sum()
    status = "[OK]" if missing == 0 else "[FAIL] INVESTIGATE!"
    print(f"\n  1. Missing values: {missing}  {status}")

    # 2. Sensor ranges
    print("\n  2. Sensor value ranges:")
    for col in SENSOR_COLS:
        lo, hi = master[col].min(), master[col].max()
        flag = ""
        if col == "volt" and (lo < 0 or hi > 300):
            flag = "  [WARN] Suspicious"
        elif col == "pressure" and lo < 0:
            flag = "  [WARN] Negative!"
        else:
            flag = "  [OK]"
        print(f"     {col:12s}  [{lo:.2f}, {hi:.2f}]{flag}")

    # 3. Time ordering
    time_ok = master.groupby("machineID")["datetime"].apply(
        lambda x: x.is_monotonic_increasing
    ).all()
    status = "[OK]" if time_ok else "[FAIL] RE-SORT!"
    print(f"\n  3. Time ordering correct: {time_ok}  {status}")

    # 4. Label columns
    print("\n  4. Label columns:")
    for col in ["failed", "failure_within_24h", "component_at_risk"]:
        if col in master.columns:
            print(f"     {col}: present  (unique: {master[col].nunique()})")
        else:
            print(f"     {col}: MISSING!")

    # 5. Shape & distribution
    print(f"\n  5. Final shape: {master.shape[0]:,} rows x {master.shape[1]} columns")
    ratio = master["failure_within_24h"].value_counts()
    denom = max(ratio.get(1, 1), 1)
    print(f"     24h-label ratio (normal : at-risk) ~ {ratio[0] // denom} : 1")

    # 6. All columns
    print(f"\n  6. Column listing:")
    for col in master.columns:
        print(f"     {col:30s}  {master[col].dtype}")

    # Save
    master.to_csv(OUTPUT_PATH, index=False)
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"\n  [DONE] Preprocessed dataset saved -> {OUTPUT_PATH}")
    print(f"     Size: {size_mb:.1f} MB")
    print(f"     Shape: {master.shape}")


# ===============================================================
#  MAIN
# ===============================================================
def main():
    print("\n" + "#" * 70)
    print("  PREDICTIVE MAINTENANCE -- PREPROCESSING PIPELINE")
    print("#" * 70)

    # Load data
    print("\n  Loading datasets...")
    telemetry = pd.read_csv(os.path.join(RAW_DATA_DIR, "PdM_telemetry.csv"))
    errors = pd.read_csv(os.path.join(RAW_DATA_DIR, "PdM_errors.csv"))
    maint = pd.read_csv(os.path.join(RAW_DATA_DIR, "PdM_maint.csv"))
    failures = pd.read_csv(os.path.join(RAW_DATA_DIR, "PdM_failures.csv"))
    machines = pd.read_csv(os.path.join(RAW_DATA_DIR, "PdM_machines.csv"))
    print("  [OK] All 5 datasets loaded")

    # Stage 1
    stage_01_understand(telemetry, errors, maint, failures, machines)

    # Stage 2
    telemetry, errors, maint, failures, machines = stage_02_types(
        telemetry, errors, maint, failures, machines
    )

    # Stage 3
    telemetry, errors, maint, failures = stage_03_sort(
        telemetry, errors, maint, failures
    )

    # Stage 4
    telemetry = stage_04_missing(telemetry, errors, maint, failures, machines)

    # Stage 5
    telemetry = stage_05_outliers(telemetry)

    # Stage 6
    telemetry = stage_06_smoothing(telemetry)

    # Stage 7
    telemetry, errors, maint, failures, machines = stage_07_duplicates(
        telemetry, errors, maint, failures, machines
    )

    # Stage 8
    stage_08_consistency(telemetry, errors, maint, failures, machines)

    # Stage 9
    master = stage_09_merge(telemetry, errors, maint, failures, machines)

    # Stage 10
    master = stage_10_labels(master)

    # Stage 11
    stage_11_imbalance(master)

    # Stage 12
    stage_12_scaling_info()

    # Stage 13
    stage_13_validate(master)

    print("\n" + "#" * 70)
    print("  PIPELINE COMPLETE [OK]")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
