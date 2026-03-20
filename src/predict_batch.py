"""
Batch Prediction Script
=======================
Feed a CSV file of engineered machine data to the trained 
XGBoost model and get failure probabilities.

Usage: 
    python src/predict_batch.py --input data/sample_input.csv --output data/predictions.csv
"""

import pandas as pd
import xgboost as xgb
import argparse
import os

PROJECT_ROOT = r"d:\HackMasters"
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "xgboost_v2_48h.json")

def main():
    parser = argparse.ArgumentParser(description="Predict machine failure probabilities from CSV.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", type=str, default="data/predictions.csv", help="Path to save predictions")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to XGBoost JSON model")
    
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found at {args.model}")
        print("Please run `python src/model_training_v2.py` first to generate it.")
        return

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found at {args.input}")
        return

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    print(f"Loading model from {args.model}...")
    model = xgb.XGBClassifier()
    model.load_model(args.model)
    
    # Get expected features by the model
    expected_features = model.get_booster().feature_names
    
    # Ensure all features exist in the input dataframe
    missing_cols = [col for col in expected_features if col not in df.columns]
    if missing_cols:
        print(f"[WARNING] Missing {len(missing_cols)} expected features. Filling with 0.")
        for col in missing_cols:
            df[col] = 0
            
    # Also we need to handle categorical casting if required, just like training
    X = df[expected_features].copy()
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = X[c].astype("category").cat.codes

    print("Running predictions...")
    probs = model.predict_proba(X)[:, 1]
    
    # Create output dataframe
    results = pd.DataFrame()
    if "datetime" in df.columns:
        results["datetime"] = df["datetime"]
    if "machineID" in df.columns:
        results["machineID"] = df["machineID"]
        
    results["Probability_of_Failure_48h"] = probs.round(4)
    results["Alert_Status"] = ["CRITICAL" if p > 0.8 else "WARNING" if p > 0.4 else "OK" for p in probs]
    
    # If the user provided the answer key, let's show it for review purposes
    if "failure_within_48h" in df.columns:
        results["Actual_Failure_Label"] = df["failure_within_48h"]
        results["Correct?"] = (results["Alert_Status"] == "CRITICAL") == (results["Actual_Failure_Label"] == 1)

    print(f"\nResults preview:\n{results.head(10).to_string()}")
    
    results.to_csv(args.output, index=False)
    print(f"\n[SUCCESS] Saved predictions to {args.output}")

if __name__ == "__main__":
    main()
