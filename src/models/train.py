from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
import mlflow

RANDOMSTATE = 42
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


def trainModel(df, targetCol):
    # Only keeps flights that are not cancelled or diverted

    dfDelay = df[(df["Cancelled"] == False) & (df["Diverted"] == False)].copy()
    dfDelay = dfDelay.dropna(subset=[targetCol]) 


    counts = dfDelay["DelayStatus"].value_counts()
    print(counts)

    print(f"Data split: \nOnTime: {counts["OnTime"]/dfDelay.shape[0]*100:.2f}%, Delayed: {counts["Delayed"]/dfDelay.shape[0]*100:.2f}% \n")


    X = dfDelay[featureCols].copy()
    y = dfDelay[targetCol].copy()

    print("Total data shape: ")
    print(X.shape, y.shape, "\n")

    XTrain, XTest, yTrain, yTest = train_test_split(X, y, train_size=0.1, test_size=0.2, random_state=RANDOMSTATE, stratify=y) # Stratify = ensures test-train split has same ratio of classes after split


    scalePosWeight = (yTrain == 0).sum() / (yTrain == 1).sum() # Used to adjust for dataset imbalance


    xgb = XGBClassifier(
        n_estimators = 100,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOMSTATE,
        n_jobs=2,
        scale_pos_weight = scalePosWeight,
        eval_metric="logloss"
    )

    # MLflow to log the model training

    with mlflow.start_run():
        xgb.fit(XTrain, yTrain)
        yPred = xgb.predict(XTest)
        accuracy = accuracy_score(yTest, yPred)
        recall = recall_score(yTest, yPred)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.xgboost.log_model(xgb, "model")

        trainDs = mlflow.data.from_pandas(dfDelay)
        mlflow.log_input(trainDs, context="training")

        print(f"Model trained.\nAccuracy: {accuracy:.2f}     Recall: {recall:.2f}")