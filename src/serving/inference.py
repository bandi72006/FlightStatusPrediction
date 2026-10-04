import pandas as pd
import mlflow
import os
import sys


def predict():
    try:
        print("Loading model")
        modelPath = os.path.join(
            os.path.dirname(__file__),
            "model",
            "m-7b08fcd1cc9b4b94b06b15a87e5bf9ac",
            "artifacts"
        )

        model = mlflow.pyfunc.load_model(modelPath)
        print("Model loading DONE")

        features = ["DepHour","ArrHour","DayOfWeek","DayofMonth","Month","Quarter","IsWeekend","Season_Summer","Season_Winter","Season_Spring","Distance","CRSElapsedTime","DistanceGroup","Airline","OriginAirportID","DestAirportID","OriginCityMarketID","DestCityMarketID","OriginWac","DestWac","OriginStateFips","DestStateFips"]
        values = [17,19,4,17,3,1,0,0,0,1,529.0,129.0,3,16,11057,11618,31057,31703,36,21,37,34]
        data = pd.DataFrame([values], columns=features)
        return model.predict(data).tolist()


    except Exception as e:
        print(f"Error failed to load model: {e}")

