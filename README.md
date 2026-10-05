# Flight Status Prediction

Using simple ML classifiers, users can input information about their flight to predict whether a flight would be delayed or not!

Trained on the [Flight Status Prediction dataaset](https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022) from Kaggle.

## Running the code

To input your data, `cd` into the web app's folder:

```
cd src/app
```

From there, run the webapp via uvicorn

```
uvicorn main:app --reload
```

Wait for your server to start up. The UI can be found in http://127.0.0.1:8000/ui/

## Machine Learning

The selected model for this classification problem was the XGBoost classifier. It performed the best compared to other tested models (LightGBM and Random Forest).

The hyperparameters of the model was finetuned using Optuna, which is the model used for inference.

## Training a model

To train your own model, you will need to have a local version of the dataset in your directory. The dataset must be in `.csv` format.

From the project root directory, run the following command:


```
python scripts/runPipeline.py --input "PATH_TO_DATASET.csv" --target DepDel15
```

A model will be trained and saved in an `mlruns` folder