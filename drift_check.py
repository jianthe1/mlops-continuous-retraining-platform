import os
import sys
import json
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 0. Version-Resilient Imports for Evidently
try:
    from evidently import Report
except (ImportError, ModuleNotFoundError):
    from evidently.report import Report

try:
    from evidently.presets import DataDriftPreset
except (ImportError, ModuleNotFoundError):
    from evidently.metric_preset import DataDriftPreset

# Helper function: Universal dictionary extractor for any Evidently version
def extract_report_dict(report):
    for method_name in ["as_dict", "to_dict", "dict"]:
        if hasattr(report, method_name) and callable(getattr(report, method_name)):
            return getattr(report, method_name)()
    if hasattr(report, "json") and callable(getattr(report, "json")):
        res = report.json()
        return json.loads(res) if isinstance(res, str) else res
    raise AttributeError("Unable to convert Evidently Report to dictionary.")

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

report_dict = extract_report_dict(drift_report)

# Safely search for dataset_drift status in the result dictionary
dataset_drift = False
metrics_list = report_dict.get("metrics", [])
for m in metrics_list:
    result = m.get("result", {})
    if "dataset_drift" in result:
        dataset_drift = bool(result["dataset_drift"])
        break

# 3. Train and Upload to MLflow / MinIO if Drift Detected
if dataset_drift:
    print("🚨 ALERT: Data drift detected! Retraining model and logging to MLflow & MinIO...")

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
    acc = accuracy_score(y_test, preds)

    # Log to MLflow & MinIO
    with mlflow.start_run(run_name="retrained_model_drift_trigger"):
        mlflow.log_metric("accuracy", acc)
        mlflow.log_param("drift_detected", True)
        
        # Log and Register Model (Uploads .pkl binary to MinIO S3 bucket)
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            registered_model_name="iris-model"
        )
        print(f"✅ Model retrained and uploaded to MLflow & MinIO! Accuracy: {acc:.4f}")

else:
    print("💚 System Healthy: No significant data drift detected. Skipping retraining.")