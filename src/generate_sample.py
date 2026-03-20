import pandas as pd
import os

PROJECT_ROOT = r"d:\HackMasters"
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "master_v2.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "sample_input.csv")

def main():
    print("Generating sample data for testing...")
    df = pd.read_csv(DATA_PATH)
    
    # Get some failures and some normal cases from the test set (latter 20%)
    train_size = int(len(df) * 0.8)
    df_test = df.iloc[train_size:]
    
    failures = df_test[df_test["failure_within_48h"] == 1].sample(5, random_state=42)
    normals = df_test[df_test["failure_within_48h"] == 0].sample(5, random_state=42)
    
    sample_df = pd.concat([failures, normals]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # We keep the target column just so the user can see if the prediction was right,
    # but the prediction script will ignore it if present.
    sample_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(sample_df)} sample rows to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
