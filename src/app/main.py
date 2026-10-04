from fastapi import FastAPI
import os
import sys
import gradio as gr

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
def getPrediction():
    try:
        result = predict()
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}

def gradioInterface(DepHour,ArrHour,DayOfWeek,DayofMonth,Month,Quarter,IsWeekend,Season_Summer,Season_Winter,Season_Spring,Distance,CRSElapsedTime,DistanceGroup,Airline,OriginAirportID,DestAirportID,OriginCityMarketID,DestCityMarketID,OriginWac,DestWac,OriginStateFips,DestStateFips):
    data = {
        "DepHour": DepHour,
        "ArrHour": ArrHour,
        "DayOfWeek": DayOfWeek,
        "DayofMonth": DayofMonth,
        "Month": Month,
        "Quarter": Quarter,
        "IsWeekend": IsWeekend,
        "Season_Summer": Season_Summer,
        "Season_Winter": Season_Winter,
        "Season_Spring": Season_Spring,
        "Distance": Distance,
        "CRSElapsedTime": CRSElapsedTime,
        "DistanceGroup": DistanceGroup,
        "Airline": Airline,
        "OriginAirportID": OriginAirportID,
        "DestAirportID": DestAirportID,
        "OriginCityMarketID": OriginCityMarketID,
        "DestCityMarketID": DestCityMarketID,
        "OriginWac": OriginWac,
        "DestWac": DestWac,
        "OriginStateFips": OriginStateFips,
        "DestStateFips": DestStateFips,
    }

    result = predict(data)
    return str(result) 


demo = gr.Interface(
    fn=gradioInterface,
    inputs = [
        gr.Number(label="DepHour"),
        gr.Number(label="ArrHour"),
        gr.Number(label="DayOfWeek"),
        gr.Number(label="DayofMonth"),
        gr.Number(label="Month"),
        gr.Number(label="Quarter"),
        gr.Number(label="IsWeekend"),
        gr.Number(label="Season_Summer"),
        gr.Number(label="Season_Winter"),
        gr.Number(label="Season_Spring"),
        gr.Number(label="Distance"),
        gr.Number(label="CRSElapsedTime"),
        gr.Number(label="DistanceGroup"),
        gr.Number(label="Airline"),
        gr.Number(label="OriginAirportID"),
        gr.Number(label="DestAirportID"),
        gr.Number(label="OriginCityMarketID"),
        gr.Number(label="DestCityMarketID"),
        gr.Number(label="OriginWac"),
        gr.Number(label="DestWac"),
        gr.Number(label="OriginStateFips"),
        gr.Number(label="DestStateFips"),

    ],
    outputs=gr.Textbox(label="Flight delay prediction", lines=2),
    title="Flight delay predictor",

    theme=gr.themes.Soft()
)

app = gr.mount_gradio_app(
    app,           # FastAPI application instance
    demo,          # Gradio interface
    path="/ui"     # URL path where Gradio will be accessible
)


