print("Importing libraries")
import os
import sys
import argparse
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Allows to append src folder
from src.data.loadData import loadData
from src.data.preprocess import processData

print("Importing DONE")

RANDOMSTATE = 42

def main(args):
    projectRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlRunsPath = f"file://{projectRoot}/mlruns"
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run():
        mlflow.log_param("model", "xgboost")
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("test_size", args.test_size)

        print("Loading data")
        df = loadData(args.input)
        print("Data loading DONE")

        print("Preprocessing and cleaning data")
        df = processData(df, args.target)
        print("Data preprocessing DONE")

        processedPath = os.path.join(projectRoot, "dataset", "flightsProcessed.csv")
        df.to_csv(processedPath, index=False)

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


        X = df[featureCols]
        target = args.target
        y = df[target]

        XTrain, XTest, yTrain, yTest = train_test_split(X, y, train_size=0.1, test_size=args.test_size, random_state=RANDOMSTATE, stratify=y) # Stratify = ensures test-train split has same ratio of classes after split

        scalePosWeight = (yTrain == 0).sum() / (yTrain == 1).sum()

        # TO-DO: Find and change hyperparameters to optimal values via optuna
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

        trainingTimeStart = time.time()
        print("Training model")
        xgb.fit(XTrain, yTrain)
        trainingTime = time.time()-trainingTimeStart
        mlflow.log_metric("train_time", trainingTime)
        print("Training model DONE")
        print(f"Model trained in: {trainingTime:.2f}s")

        print("Evaluating model")

        predTimeStart = time.time()
        proba = xgb.predict_proba(XTest)[:,1]
        yPred = (proba >= args.threshold).astype(int)

        predTime = time.time() - predTimeStart
        mlflow.log_metric("pred_time", predTime)


        precision = precision_score(yTest, yPred)
        recall = recall_score(yTest, yPred)
        f1 = f1_score(yTest, yPred)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall) 
        mlflow.log_metric("f1", f1)

        print("Model performance: ")
        print(f"Precision: {precision:2f}")
        print(f"Recall: {recall:2f}")
        print(f"f1: {f1:2f}")

        print("Saving model to MLflow")

        mlflow.sklearn.log_model(xgb, 
            name="model",
            skops_trusted_types=[
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier"
            ]
        )
        
        print("Model saving DONE")

        print("Summary: ")
        print(f"Training time: {trainingTime:.2f}s")
        print(f"Inference time: {predTime:.2f}s")

        print("Classification report:")
        print(classification_report(yTest, yPred, digits=3))

        print("Pipeline complete...")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run ML pipeline")
    p.add_argument("--input", type=str, required=True,
                   help="path to CSV (e.g., ..\data\Combined_Flights_2022.csv")
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Flight delay")
    p.add_argument("--mlflow_uri", type=str, default=None,
                    help="override MLflow tracking URI, else uses project_root/mlruns")

    args = p.parse_args()
    main(args)


# To run pipline:
# python scripts/runPipeline.py --input dataset\Combined_Flights_2022.csv --target DepDel15