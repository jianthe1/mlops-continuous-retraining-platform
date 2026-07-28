import os
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="MLOps Production Serving API")

# Set MLflow Tracking URI
MLFLOW_SERVER = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
mlflow.set_tracking_uri(MLFLOW_SERVER)

# Load the latest production model on startup
MODEL_NAME = "production_churn_model"
MODEL_STAGE = "Production"
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        print(f"Warning: Could not load production model: {e}")

class FeaturesPayload(BaseModel):
    features: list[float]

# SRE Probes
@app.get("/healthz")
def health_check():
    return {"status": "healthy"}

@app.get("/readyz")
def readiness_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}

@app.post("/predict")
def predict(payload: FeaturesPayload):
    if not model:
        raise HTTPException(status_code=500, detail="Model unavailable")
    prediction = model.predict([payload.features])
    return {"prediction": int(prediction[0])}