import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def processData(df, targetCol):
    postFlightCols = [
        "DepTime", "DepDelay", "DepDelayMinutes",
        "ArrTime",                                    
        "ArrDelayMinutes",                            
        "ActualElapsedTime",                          
        "WheelsOff", "WheelsOn",                      
        "ArrivalDelayGroups",                         
        "DivAirportLandings"
    ]

    redundantCols = ["DOT_ID_Marketing_Airline", "IATA_Code_Marketing_Airline",
        "DOT_ID_Operating_Airline", "IATA_Code_Operating_Airline",
        "OriginAirportSeqID", "OriginCityName", "OriginStateName",
        "DestAirportSeqID", "DestCityName", "DestStateName",
        "Year"]

    dropCols = postFlightCols + redundantCols

    df.drop(columns=dropCols, inplace=True) # inplace=True -> applies changes to actual df

    # Handle missing values

    df.fillna({"CRSElapsedTime":df["CRSElapsedTime"].median()}, inplace=True)
    df.fillna({"Tail_Number": "None"}, inplace=True)
    df.fillna({"Operated_or_Branded_Code_Share_Partners": "None"}, inplace=True)
    df.fillna({"ArrTimeBlk": df["ArrTimeBlk"].mode()[0]}, inplace=True)
    df.fillna({"CRSDepTime": 0}, inplace=True)
    df.fillna({"CRSArrTime": 0}, inplace=True) 

    # Columns to engineer
    viewCols = ["CRSDepTime", "CRSArrTime", "DayOfWeek", "Month"]

    df["DepHour"] = (df["CRSDepTime"].astype(int) // 100).clip(0,23)
    df["ArrHour"] = (df["CRSArrTime"].astype(int) // 100).clip(0,23)
    df["IsWeekend"] = (df["DayOfWeek"] >= 6).astype(int)

    seasonMap = {12: 0,1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
                6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}
    df["Season"] = df["Month"].map(seasonMap)
    if "Season_Summer" not in df.columns and "Season_Spring" not in df.columns and "Season_Winter" not in df.columns: # No fall season in dataset
        df = pd.get_dummies(df, columns=["Season"]) # One-hot encodes feature

    le = LabelEncoder()
    df["Airline"] = le.fit_transform(df["Airline"]) # Encodes airline to a digit between 0 and numberOfAirlines-1

    # Y-values

    conditions = [df["Cancelled"] == True, df["Diverted"] == True, df["DepDel15"] == 1.0]
    choices = ["Cancelled", "Diverted", "Delayed"]
    df["DelayStatus"] = np.select(conditions, choices, default="OnTime")

    # Convert boolean to ints

    boolCols = df.select_dtypes(include="bool").columns
    df[boolCols] = df[boolCols].astype(int)

    df = df.dropna(subset=[targetCol]) 

    featureCols = [
        'DepHour', 'ArrHour', 'DayOfWeek', 'DayofMonth', 'Month', 'Quarter',
        'IsWeekend', 'Season_Summer', 'Season_Winter', 'Season_Spring',
        'Distance', 'CRSElapsedTime', 'DistanceGroup',
        'Airline', 
        'OriginAirportID', 'DestAirportID',
        'OriginCityMarketID', 'DestCityMarketID',
        'OriginWac', 'DestWac',
        'OriginStateFips', 'DestStateFips'
    ]

    return df[featureCols + [targetCol]]
