import pandas as pd
import mlflow

def predict():
    try:
        model = mlflow.pyfunc.load_model("/app/model")
        print("Model loaded DONE")

    except Exception as e:
        print(f"Error failed to load model: {e}")

