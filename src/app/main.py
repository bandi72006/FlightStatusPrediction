from fastapi import FastAPI
import os
import sys
import gradio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.serving.inference import predict

app = FastAPI(
    title = "Flight delay prediction API",
    description = "ML API for predicting whether a flight will be delayed",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
def getPrediction(data):
    try:
        result = predict(data.dict())
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}