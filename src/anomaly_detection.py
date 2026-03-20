"""
Anomaly Detection Pipeline - Predictive Maintenance
===================================================
Input  : master_featured.csv (from feature engineering)
Output : master_anomaly.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score
import os
import warnings

# Suppress TF warnings if tensorflow is missing, use a simple autoencoder fallback
try:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    has_tf = True
except ImportError:
    has_tf = False
    print("TensorFlow not found, using PCA as a fallback for Autoencoder substitute.")
    from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "plots")

INPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "master_featured.csv")
OUTPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "master_anomaly.csv")

def main():
    print("\n" + "#" * 60)
    print("  ANOMALY DETECTION PIPELINE")
    print("#" * 60)

    # STEP 1
    print("\n  STEP 1 -- Prepare Feature Dataset")
    df = pd.read_csv(INPUT_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])

    DROP_COLS = ["datetime", "machineID", "failure", "failed",
                 "failure_within_24h", "component_at_risk",
                 "failure_comp1", "failure_comp2", "failure_comp3", "failure_comp4",
                 "model", "age_group", "risk_level"]

    feature_cols = [c for c in df.columns if c not in DROP_COLS 
                    and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
    print(f"  Selected {len(feature_cols)} numeric features.")
    
    X = df[feature_cols].copy()
    y_true = df["failure_within_24h"].copy()

    # STEP 2
    print("\n  STEP 2 -- Feature Scaling (StandardScaler)")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)

    # STEP 3A
    print("\n  STEP 3A -- Isolation Forest")
    iso_forest = IsolationForest(n_estimators=100, contamination=0.02, random_state=42, n_jobs=-1)
    iso_forest.fit(X_scaled)
    df["iso_score"] = iso_forest.decision_function(X_scaled)
    df["iso_anomaly"] = iso_forest.predict(X_scaled)

    # STEP 3B
    print("\n  STEP 3B -- One-Class SVM (Subsampled)")
    sample_idx = X_scaled.sample(n=min(30000, len(X_scaled)), random_state=42).index
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.02)
    ocsvm.fit(X_scaled.loc[sample_idx])
    df["ocsvm_score"] = ocsvm.decision_function(X_scaled)
    df["ocsvm_anomaly"] = ocsvm.predict(X_scaled)

    # STEP 3C
    print("\n  STEP 3C -- Autoencoder / Reconstruction")
    normal_mask = df["failure_within_24h"] == 0
    if has_tf:
        inp = Input(shape=(len(feature_cols),))
        encoded = Dense(64, activation="relu")(inp)
        encoded = Dense(32, activation="relu")(encoded)
        latent  = Dense(16, activation="relu")(encoded)
        decoded = Dense(32, activation="relu")(latent)
        decoded = Dense(64, activation="relu")(decoded)
        out  = Dense(len(feature_cols), activation="linear")(decoded)
        
        ae = Model(inp, out)
        ae.compile(optimizer="adam", loss="mse")
        
        ae.fit(X_scaled[normal_mask], X_scaled[normal_mask],
               epochs=10, batch_size=512, validation_split=0.1,
               callbacks=[EarlyStopping(patience=3)], verbose=0)
        
        recon = ae.predict(X_scaled, batch_size=1024)
        df["ae_recon_error"] = np.mean((X_scaled.values - recon) ** 2, axis=1)
    else:
        # PCA fallback
        pca = PCA(n_components=16, random_state=42)
        pca.fit(X_scaled[normal_mask])
        recon = pca.inverse_transform(pca.transform(X_scaled))
        df["ae_recon_error"] = np.mean((X_scaled.values - recon) ** 2, axis=1)

    threshold = df.loc[normal_mask, "ae_recon_error"].quantile(0.98)
    df["ae_anomaly"] = np.where(df["ae_recon_error"] > threshold, -1, 1)

    # STEP 4
    print("\n  STEP 4 -- Generate Anomaly Scores")
    df["anomaly_votes"] = ((df["iso_anomaly"] == -1).astype(int) + 
                           (df["ocsvm_anomaly"] == -1).astype(int) + 
                           (df["ae_anomaly"] == -1).astype(int))
    df["ensemble_anomaly"] = np.where(df["anomaly_votes"] >= 2, -1, 1)
    
    # STEP 5
    print("\n  STEP 5 -- Evaluate Anomaly Detection")
    y_pred = (df["ensemble_anomaly"] == -1).astype(int)
    print("Classification Report (Anomaly vs Failure):")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Failure Window"]))
    print(f"ROC-AUC (using ISO score): {roc_auc_score(y_true, -df['iso_score']):.4f}")

    # STEP 6
    print("\n  STEP 6 -- Visualize Anomalies")
    machine = df[df["machineID"] == 1].copy()
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    for ax, col in zip(axes, ["volt", "rotate", "pressure", "vibration"]):
        ax.plot(machine["datetime"], machine[col], linewidth=0.5, alpha=0.7)
        anoms = machine[machine["ensemble_anomaly"] == -1]
        ax.scatter(anoms["datetime"], anoms[col], c="red", s=8, label="Anomaly")
        ax.set_ylabel(col)
        ax.legend(loc="upper right")
    axes[0].set_title("Machine 1 - Sensor Readings with Anomalies")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "anomaly_visualization.png"), dpi=120)
    plt.close()

    # STEP 7
    print("\n  STEP 7 -- Build Early Warning System")
    def assess_risk(row):
        score = 0
        if row["ensemble_anomaly"] == -1: score += 3
        if row.get("vibration_trend", 0) > 0: score += 1
        if row.get("pressure_trend", 0) < 0: score += 1
        if row.get("error_count_24h", 0) > 2: score += 2
        if score >= 5: return "CRITICAL"
        elif score >= 3: return "HIGH"
        elif score >= 1: return "MEDIUM"
        return "LOW"
    df["risk_level"] = df.apply(assess_risk, axis=1)
    print("Risk levels:")
    print(df["risk_level"].value_counts())

    # FINAL
    print("\n  FINAL STEP -- Validate & Save")
    print(f"Anomaly rate: {(df['ensemble_anomaly'] == -1).mean() * 100:.2f}%")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved -> {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
