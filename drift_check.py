import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
import requests

# Load Reference (Training) and Current (Production) Data
reference_df = pd.read_csv("data/reference_train.csv")
current_df = pd.read_csv("data/production_logs.csv")

# Generate Drift Report
drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(reference_data=reference_df, current_data=current_df)

report_dict = drift_report.as_dict()
dataset_drift = report_dict["metrics"][0]["result"]["dataset_drift"]

# SRE Logic: Trigger Automated Retraining Alert if Data Drift Detected
if dataset_drift:
    print("ALERT: Data drift detected above threshold (>20%)!")
    # Send Webhook to GitHub Actions / Airflow to trigger retraining pipeline
    requests.post(
        "https://api.github.com/repos/YOUR_REPO/dispatches",
        json={"event_type": "trigger-retrain"},
        headers={"Authorization": "token YOUR_GITHUB_TOKEN"}
    )
else:
    print("System Healthy: No significant drift detected.")