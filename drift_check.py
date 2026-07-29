import os
import sys
import pandas as pd
import requests

# 0. Version-Resilient Import for Evidently
from evidently.report import Report
try:
    from evidently.metric_preset import DataDriftPreset
except ModuleNotFoundError:
    try:
        from evidently.presets import DataDriftPreset
    except ModuleNotFoundError:
        from evidently.metric_preset import DataDriftPreset

# 1. Fetch Environment Variables (Accepts PAT_TOKEN or GITHUB_TOKEN)
PAT_TOKEN = os.getenv("PAT_TOKEN") or os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO", "jianthe1/mlops-continuous-retraining-platform")

if not PAT_TOKEN:
    print("⚠️ Error: Neither PAT_TOKEN nor GITHUB_TOKEN environment variable is set.")
    sys.exit(1)

# 2. Check File Existence & Load Data
ref_path = "data/reference_train.csv"
prod_path = "data/production_logs.csv"

if not os.path.exists(ref_path) or not os.path.exists(prod_path):
    print(f"⚠️ Error: Missing dataset files. Ensure '{ref_path}' and '{prod_path}' exist.")
    sys.exit(1)

reference_df = pd.read_csv(ref_path)
current_df = pd.read_csv(prod_path)

# 3. Calculate Drift Report (>20% drifted features threshold)
print("🔍 Calculating data drift metrics...")
drift_report = Report(metrics=[DataDriftPreset(drift_share=0.2)])
drift_report.run(reference_data=reference_df, current_data=current_df)

report_dict = drift_report.as_dict()
dataset_drift = report_dict["metrics"][0]["result"]["dataset_drift"]

# 4. Trigger Retraining via GitHub Actions Dispatch
if dataset_drift:
    print("🚨 ALERT: Data drift detected (>20%)! Dispatching trigger to GitHub Actions...")
    
    url = f"https://api.github.com/repos/{GITHUB_REPO}/dispatches"
    headers = {
        "Authorization": f"Bearer {PAT_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    payload = {"event_type": "trigger-retrain"}

    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 204:
        print("✅ Retraining pipeline triggered successfully!")
    else:
        print(f"❌ Failed to trigger workflow: {response.status_code} - {response.text}")
else:
    print("💚 System Healthy: No significant data drift detected.")