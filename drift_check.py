import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# 1. Fetch File Paths
ref_path = "data/reference_train.csv"
prod_path = "data/production_logs.csv"

if not os.path.exists(ref_path) or not os.path.exists(prod_path):
    print(f"⚠️ Error: Missing dataset files. Ensure '{ref_path}' and '{prod_path}' exist.")
    sys.exit(1)

reference_df = pd.read_csv(ref_path)
current_df = pd.read_csv(prod_path)

# 2. Calculate Drift Report (>20% drifted features threshold)
print("🔍 Calculating data drift metrics...")
drift_report = Report(metrics=[DataDriftPreset(drift_share=0.2)])
drift_report.run(reference_data=reference_df, current_data=current_df)

report_dict = drift_report.as_dict()
dataset_drift = report_dict["metrics"][0]["result"]["dataset_drift"]

# 3. Train and Upload to MLflow / MinIO if Drift Detected
if dataset_drift:
    print("🚨 ALERT: Data drift detected! Retraining model...")

    # Configure MLflow Tracking & MinIO S3
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.testermy-apps.duckdns.org")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("continuous-retraining")

    # Combine production and reference data for retraining
    full_df = pd.concat([reference_df, current_df], ignore_index=True)
    X = full_df.drop(columns=["target"])
    y = full_df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = float(accuracy_score(y_test, preds))

    # ---------------------------------------------------------
    # 🛡️ MLOPS GUARDRAIL CHECK
    # ---------------------------------------------------------
    MIN_ACCURACY_FLOOR = 0.85  # Minimum acceptable model accuracy (85%)

    if acc < MIN_ACCURACY_FLOOR:
        print(f"\n🚨 GUARDRAIL BLOCKED DEPLOYMENT!")
        print(f"❌ Retrained model accuracy ({acc:.4f}) fell below required floor ({MIN_ACCURACY_FLOOR}).")
        print("⛔ Aborting model registration and artifact upload to MinIO.")
        
        # Log failure metadata for debugging without saving model artifacts
        with mlflow.start_run(run_name="rejected_model_guardrail"):
            mlflow.log_metric("accuracy", acc)
            mlflow.set_tag("guardrail_status", "REJECTED")
            mlflow.set_tag("failure_reason", "accuracy_below_floor")

        sys.exit(1)  # Stop workflow & trigger alert in GitHub Actions

    # ---------------------------------------------------------
    # Log to MLflow & MinIO if Guardrail Passed
    # ---------------------------------------------------------
    print(f"✅ Model passed quality guardrails! Accuracy: {acc:.4f}")

    with mlflow.start_run(run_name="retrained_model_drift_trigger"):
        mlflow.log_metric("accuracy", acc)
        mlflow.log_param("drift_detected", True)
        mlflow.set_tag("guardrail_status", "PASSED")
        
        # Log and Register Model
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            registered_model_name="iris-model"
        )
        print("🎉 Retrained model successfully logged to MLflow & uploaded to MinIO!")

else:
    print("💚 System Healthy: No significant data drift detected. Skipping retraining.")