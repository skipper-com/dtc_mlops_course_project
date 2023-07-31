import os
import json
import pickle
import pathlib
from datetime import date

import s3fs
import numpy as np
import scipy
import mlflow
import pandas as pd
import sklearn
import xgboost as xgb
from prefect import flow, task
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error
from prefect.artifacts import create_markdown_artifact
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

s3_bucket = os.getenv("S3_BUCKET")


@task(retries=3, retry_delay_seconds=2)
def read_data(filename: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    df = pd.read_csv(f"s3://{s3_bucket}/data/{filename}", low_memory=False)
    # df = pd.read_csv(f"data/{filename}", low_memory=False)

    return df


@task
def process_features(
    df: pd.DataFrame,
) -> tuple(
    [
        scipy.sparse._csr.csr_matrix,
        scipy.sparse._csr.csr_matrix,
        np.ndarray,
        np.ndarray,
        sklearn.feature_extraction.DictVectorizer,
    ]
):
    """Process features, compose vectors"""
    categorical = ["neighbourhood", "room_type"]
    numerical = ["availability_365"]
    target = ["price"]
    df = df[categorical + numerical + target]
    df[categorical] = df[categorical].astype(str)
    df = df[df.price <= df.price.quantile(q=0.95)]

    df_train, df_val = train_test_split(df, test_size=0.2)

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    y_train = df_train[target].values
    y_val = df_val[target].values

    return X_train, X_val, y_train, y_val, dv


@task(log_prints=True)
def train_best_model(
    X_train: scipy.sparse._csr.csr_matrix,
    X_val: scipy.sparse._csr.csr_matrix,
    y_train: np.ndarray,
    y_val: np.ndarray,
    dv: sklearn.feature_extraction.DictVectorizer,
) -> None:
    """Train a model with best hyperparams and write everything out"""

    with mlflow.start_run():
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            "learning_rate": 0.2991109761625923,
            "max_depth": 4,
            "min_child_weight": 10.299631482448909,
            "objective": "reg:linear",
            "reg_alpha": 0.030220138602381437,
            "reg_lambda": 0.0758045390119753,
            "seed": 42,
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, "validation")],
            early_stopping_rounds=20,
        )

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        with open("models/lin_reg.bin", "wb") as f_out:
            pickle.dump((dv, booster), f_out)
        mlflow.log_artifact(
            local_path="models/lin_reg.bin", artifact_path="models_pickle"
        )

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        markdown__rmse_report = f"""# RMSE Report

        ## Summary

        Duration Prediction

        ## RMSE XGBoost Model

        | Date    | RMSE |
        |:----------|-------:|
        | {date.today()} | {rmse:.2f} |

        """
        create_markdown_artifact(
            key="airbnb-price-report", markdown=markdown__rmse_report
        )
        mlflow.search_runs(filter_string="run_name='my_run'")["run_id"]

        experiment_name = "airbnb-price-prod"
        current_experiment = dict(mlflow.get_experiment_by_name(experiment_name))
        experiment_id = current_experiment["experiment_id"]

        rmse = mlflow.search_runs([experiment_id], order_by=["metrics.rmse ASC"])
        best_run_id = rmse.loc[0, "run_id"]

        data = {"run_id": best_run_id}
        data = json.dumps(data)

        with open("models/run_id.json", "w", encoding="utf-8") as f_out:
            f_out.write(data)

    return None


@flow
def airbnb_flow(df_path: str = "NYC-Airbnb-2023.csv") -> None:
    """The main training pipeline"""

    # MLflow settings
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("airbnb-price-prod")

    # Load
    df = read_data(df_path)

    # Transform
    X_train, X_val, y_train, y_val, dv = process_features(df)

    # Train
    train_best_model(X_train, X_val, y_train, y_val, dv)


if __name__ == "__main__":
    airbnb_flow()
