"""
Live Inference Example - Predictive Maintenance (V2)
=====================================================
Shows how to load a saved XGBoost model and make predictions
on a new batch of data using a historical buffer.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import os
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = r"d:\HackMasters"
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "xgboost_v2_48h.json")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

def load_or_train_model():
    """Returns a trained XGBoost model."""
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}...")
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        return model
    else:
        print("Model file not found. Please add `model.save_model('models/xgboost_v2_48h.json')` to your training script.")
        return None

def engineer_inference_features(historical_buffer_df):
    """
    Applies the V2 feature engineering logic to the historical buffer.
    historical_buffer_df must contain the last 72+ hours of merged
    telemetry, error, and maintenance data for a specific machine.
    """
    df = historical_buffer_df.copy()
    df = df.sort_values("datetime").reset_index(drop=True)
    
    # Example: Re-calculate just the core rolling features needed for the FINAL row
    df['vibration_rmean_24'] = df['vibration'].rolling(24, min_periods=1).mean()
    df['pressure_rmean_24'] = df['pressure'].rolling(24, min_periods=1).mean()
    
    # EMA
    df['volt_ema_24'] = df['volt'].ewm(span=24).mean()
    
    # Derivatives
    df['vibration_vel'] = df['vibration'].diff().fillna(0)
    
    # Interactions
    df['combined_sensor_index'] = (df['vibration_rmean_24'] + df['pressure_rmean_24']) / (df.get('age', 1) + 1)
    
    # Lags (Only possible if buffer is at least 72 rows/hours long)
    for lag in [24, 48, 72]:
        for col in ['volt', 'rotate', 'pressure', 'vibration']:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag).fillna(method='bfill')

    # Add remaining logic matching feature_engineering_v2.py exactly...
    # (For a true production pipeline, extracting the logic into shared functions is recommended).
    
    # We only care about predicting the state of the VERY LAST row (the current time)
    return df.iloc[[-1]]

def predict_new_data(model, extracted_feature_row):
    """Takes a single processed row and outputs the probability of failure."""
    # Ensure columns match training data order
    features = model.get_booster().feature_names
    
    # Add dummy columns if missing (in a real pipeline, ensure exact matching)
    for col in features:
        if col not in extracted_feature_row.columns:
            extracted_feature_row[col] = 0
            
    X_predict = extracted_feature_row[features]
    
    # Predict
    prob = model.predict_proba(X_predict)[0, 1]
    prediction = 1 if prob > 0.8 else 0  # 80% threshold for High Alert
    
    return prob, prediction

def main():
    print("\n" + "=" * 50)
    print("  LIVE INFERENCE SIMULATION")
    print("=" * 50)
    
    # 1. Load model
    model = load_or_train_model()
    if not model:
        return

    # 2. Simulate "New Data" arriving by grabbing a slice from the processed dataset
    # In reality, this comes from a database query: SELECT * FROM sensor_data WHERE machine_id=1 LIMIT 72
    print("Fetching last 72 hours of data for Machine 1 from buffer...")
    df_history = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "master_v2.csv"))
    buffer = df_history[df_history["machineID"] == 1].tail(75).copy()
    
    current_time = buffer.iloc[-1]["datetime"]
    print(f"Current Time: {current_time}")

    # 3. Predict on the last row
    features = model.get_booster().feature_names
    X_predict = buffer.iloc[[-1]][features]
    
    prob_48h = model.predict_proba(X_predict)[0, 1]
    
    print("\n  [INFERENCE RESULT]")
    print(f"  Machine 1 Status at {current_time}:")
    print(f"  Probability of Failure in next 48h: {prob_48h * 100:.2f}%")
    
    if prob_48h > 0.80:
        print("  --> 🚨 CRITICAL ALERT: Schedule Maintenance Immediately!")
    elif prob_48h > 0.40:
        print("  --> ⚠️ WARNING: Monitor Closely.")
    else:
        print("  --> ✅ STATUS: Healthy.")

if __name__ == "__main__":
    main()
