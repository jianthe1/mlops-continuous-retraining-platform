import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="MLOps Production Serving API")

# This line goes HERE inside main.py
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print("Model loaded successfully!")
        else:
            print(f"Error: {MODEL_PATH} not found.")
    except Exception as e:
        print(f"Error loading model: {e}")

class FeaturesPayload(BaseModel):
    features: list[float]

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
    try:
        prediction = model.predict([payload.features])
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))