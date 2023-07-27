import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import mlflow
from sklearn.model_selection import train_test_split
import os
from mlflow.tracking import MlflowClient
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.svm import LinearSVR
import s3fs


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("airbnb-price-experiment")
client = MlflowClient("http://127.0.0.1:5000")


def read_data(filename: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    #df = pd.read_csv(f"s3://mlops-course-project/data/{filename}", low_memory=False)
    df = pd.read_csv(f"data/{filename}", low_memory=False)
    return df


def process_features(
    df: pd.DataFrame, categorical: list, numerical: list, target: list
) -> pd.DataFrame:
    """Extract and pre-process some features"""
    df = df[categorical + numerical + target]
    df[categorical] = df[categorical].astype(str)
    df = df[df.price <= df.price.quantile(q=0.95)]

    return df


def train_model(
    df_train: pd.DataFrame,
    lr: object,
    dv: object,
    categorical: list,
    numerical: list,
    target: list,
) -> float:
    """learn model and get MSE on train dataset"""
    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)
    y_train = df_train[target]
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    return mean_squared_error(y_train, y_pred, squared=False)


def validate_model(
    df_val: pd.DataFrame,
    lr: object,
    dv: object,
    categorical: list,
    numerical: list,
    target: list,
) -> float:
    """get MSE on validaton dataset"""
    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)
    y_val = df_val[target]
    y_pred = lr.predict(X_val)
    y_pred = lr.predict(X_val)
    return mean_squared_error(y_val, y_pred, squared=False)


def test_model(
    df_test: pd.DataFrame,
    lr: object,
    dv: object,
    categorical: list,
    numerical: list,
    target: list,
) -> float:
    """get MSE on test dataset"""
    tests_dicts = df_test[categorical + numerical].to_dict(orient="records")
    X_test = dv.transform(tests_dicts)
    y_test = df_test[target]
    y_pred = lr.predict(X_test)
    return mean_squared_error(y_test, y_pred, squared=False)


def log_xgb_experiment() -> dict:
    """"""
    search_space = {
        "max_depth": scope.int(hp.quniform("max_depth", 4, 100, 1)),
        "learning_rate": hp.loguniform("learning_rate", -3, 0),
        "reg_alpha": hp.loguniform("reg_alpha", -5, -1),
        "reg_lambda": hp.loguniform("reg_lambda", -6, -1),
        "min_child_weight": hp.loguniform("min_child_weight", -1, 3),
        "objective": "reg:linear",
        "seed": 42,
    }

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=Trials(),
    )

    return best_result


def objective(params) -> dict:
    """ """
    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, "validation")],
            early_stopping_rounds=50,
        )
        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

    return {"loss": rmse, "status": STATUS_OK}


def log_models_experiments():
    """"""
    for model_class in (
        RandomForestRegressor,
        GradientBoostingRegressor,
        ExtraTreesRegressor,
        LinearSVR,
    ):
        with mlflow.start_run():
            mlflow.log_param("data-path", "./data/NYC-Airbnb-2023.csv")
            mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
            mlmodel = model_class()
            mlmodel.fit(X_train, y_train)
            y_pred = mlmodel.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)


def price_prediction() -> None:
    """predict baseline price"""
    df = read_data("NYC-Airbnb-2023.csv")
    print("data successfully read")

    categorical = ["neighbourhood", "room_type"]
    numerical = ["availability_365"]
    target = ["price"]

    df = process_features(df, categorical, numerical, target)
    print("features successfully processed")

    lr = LinearRegression()
    dv = DictVectorizer()

    mlflow.sklearn.autolog()

    df_train, df_remain = train_test_split(df, test_size=0.3)
    df_val, df_test = train_test_split(df_remain, test_size=0.3)

    mse_train = train_model(df_train, lr, dv, categorical, numerical, target)
    mse_val = train_model(df_val, lr, dv, categorical, numerical, target)
    mse_test = train_model(df_test, lr, dv, categorical, numerical, target)
    print("baseline model successfully trained")

    dump_model(lr, dv)
    print("baseline model successfully dumped")

    mlflow.sklearn.autolog()

    df_train, df_val = train_test_split(df, test_size=0.3)

    global X_train
    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    global X_val
    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    global y_train
    y_train = df_train[target]

    global y_val
    y_val = df_val[target]

    global train
    train = xgb.DMatrix(X_train, label=y_train)
    global valid
    valid = xgb.DMatrix(X_val, label=y_val)

    best_result = log_xgb_experiment()
    print("XGBoost model successfully trained")
    print(f"best result = {best_result}")

    log_models_experiments()
    print("Different models successfully trained")


def dump_model(lr: object, dv: object) -> None:
    """dump model and vectors"""
    with open("models/lin_reg.bin", "wb") as f_out:
        pickle.dump((dv, lr), f_out)

    with open("models/preprocessor.b", "wb") as f_out:
        pickle.dump(dv, f_out)


if __name__ == "__main__":
    price_prediction()
