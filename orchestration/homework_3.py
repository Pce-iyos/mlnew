
from prefect import flow, task
import pandas as pd
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import mlflow
import mlflow.sklearn

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

Path("models").mkdir(exist_ok=True)

@task(name="Load & preprocess data", log_prints=True)
def read_data(path: str):
    df = pd.read_parquet(path)
    print(f"Raw records: {df.shape}")

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    df[['PULocationID', 'DOLocationID']] = df[['PULocationID', 'DOLocationID']].astype(str)
    print(f"Filtered records: {df.shape}")
    return df

@task(name="Feature engineering")
def create_features(df: pd.DataFrame, dv: DictVectorizer = None):
    features = ["PULocationID", "DOLocationID", "trip_distance"]
    dicts = df[features].to_dict(orient="records")

    if dv is None:
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv

@task(name="Train & log model", log_prints=True)
def train_and_log(X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        # Save DictVectorizer
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="models_lr")

        print(f"Intercept of the model: {model.intercept_}")
        return model.intercept_

@flow(name="Train Linear Model on Yellow Taxi Data ")
def train_prefect_pipeline(parquet_path: str = "/home/pce/Downloads/yellow_tripdata_2023-03.parquet"):
    df = read_data(parquet_path)

    # Split
    df_train, df_val = train_test_split(df, test_size=0.3, random_state=42)

    X_train, dv = create_features(df_train)
    X_val, _ = create_features(df_val, dv)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values

    intercept = train_and_log(X_train, y_train, X_val, y_val, dv)
    print(f" Final Intercept: {intercept:.2f}")

if __name__ == "__main__":
    train_prefect_pipeline()
