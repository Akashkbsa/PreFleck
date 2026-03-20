"""
Advanced Feature Engineering (V2) - Predictive Maintenance
===========================================================
Focus: Long-term prediction (48h+), Reducing Shortcut Dependency, 
       Capturing Temporal Interaction and Rate of Change.
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = r"d:\HackMasters"
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "master_anomaly.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "master_v2.csv")

def main():
    print("\n" + "#" * 70)
    print("  FEATURE ENGINEERING REDESIGN (V2)")
    print("#" * 70)

    # 1. Load data
    df = pd.read_csv(INPUT_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["machineID", "datetime"]).reset_index(drop=True)

    print("\n  STEP 1 -- Multi-Horizon Labelling (48h)")
    # Re-calculate failure windows for 48h
    df['failure_within_48h'] = 0
    failure_indices = df[df['failed'] == 1].index
    for idx in failure_indices:
        machine = df.loc[idx, 'machineID']
        end_time = df.loc[idx, 'datetime']
        start_time = end_time - pd.Timedelta(hours=48)
        # Apply mask for the same machine
        mask = (df['machineID'] == machine) & (df['datetime'] >= start_time) & (df['datetime'] < end_time)
        df.loc[mask, 'failure_within_48h'] = 1
    
    print("\n  STEP 2 -- Weaken Shortcuts (Lagging & Long Windows)")
    # Lag error counts to prevent immediate "spike" leakage for 48h prediction
    df['error_count_lag_24'] = df.groupby('machineID')['error_count_24h'].shift(24).fillna(0)
    # 7-day long window for baseline error level
    df['error_count_168h'] = df.groupby('machineID')['error_count_24h'].transform(lambda x: x.rolling(168, min_periods=1).sum())

    print("\n  STEP 3 -- Interaction Features")
    # Capturing physical stress relationships
    df['vibration_pressure_ratio'] = df['vibration'] / (df['pressure'] + 1)
    df['energy_stress'] = df['volt'] * df['rotate']
    # Normalizing sensor interaction
    df['combined_sensor_index'] = (df['vibration_rmean_24'] + df['pressure_rmean_24']) / (df['age'] + 1)

    print("\n  STEP 4 -- Time-Series Trend & EMA")
    # Exponential weighted moving average for smoother signal
    df['volt_ema_24'] = df.groupby('machineID')['volt'].transform(lambda x: x.ewm(span=24).mean())
    # Slope of vibration
    df['vibration_slope_6h'] = df.groupby('machineID')['vibration_rmean_6'].diff().fillna(0)

    print("\n  STEP 5 -- Rate of Change (Derivatives)")
    # Vibration velocity and acceleration
    df['vibration_vel'] = df.groupby('machineID')['vibration'].diff().fillna(0)
    df['vibration_accel'] = df.groupby('machineID')['vibration_vel'].diff().fillna(0)

    print("\n  STEP 6 -- Long-Term Memory")
    # Calculate maint_any first
    maint_cols = ['maint_comp1', 'maint_comp2', 'maint_comp3', 'maint_comp4']
    df['maint_any'] = df[maint_cols].sum(axis=1) > 0
    # Cumulative errors over machine lifetime
    df['cum_errors'] = df.groupby('machineID')['failed'].cumsum()
    # Time since last maintenance (already exists/improved)
    df['maint_freq_30d'] = df.groupby('machineID')['maint_any'].transform(lambda x: x.rolling(720, min_periods=1).sum())

    print("\n  STEP 7 -- Multi-Horizon Lags (24h, 48h, 72h)")
    # Helping model remember state at the prediction horizon boundaries
    for lag in [24, 48, 72]:
        for col in ['volt', 'rotate', 'pressure', 'vibration']:
            df[f'{col}_lag_{lag}'] = df.groupby('machineID')[col].shift(lag).fillna(method='bfill')

    print("\n  STEP 8 -- Feature Selection & Reduction")
    # Identify high correlation features and redundant columns
    # We will exclude failure_within_24h to avoid accidental leakage in new model
    X_sample = df.drop(columns=['machineID', 'datetime', 'failure', 'failed', 'component_at_risk', 
                               'failure_comp1', 'failure_comp2', 'failure_comp3', 'failure_comp4', 
                               'model', 'age_group', 'risk_level'])
    
    corr_matrix = X_sample.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.98)] # Extreme redundancy
    
    print(f"  Dropping {len(to_drop)} redundant features: {to_drop}")
    df_final = df.drop(columns=to_drop)

    print("\n  STEP 9 -- Final Validation & Save")
    print(f"  Final columns count: {len(df_final.columns)}")
    print(f"  Target distribution (48h): \n{df_final['failure_within_48h'].value_counts(normalize=True)}")
    
    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  [SUCCESS] Saved V2 dataset -> {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
