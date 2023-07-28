import pickle

import s3fs
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer


def read_data(filename: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    # df = pd.read_csv(f"s3://mlops-course-project/data/{filename}", low_memory=False)
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

    df_train, df_remain = train_test_split(df, test_size=0.3)
    df_val, df_test = train_test_split(df_remain, test_size=0.3)

    mse_train = train_model(df_train, lr, dv, categorical, numerical, target)
    mse_val = train_model(df_val, lr, dv, categorical, numerical, target)
    mse_test = train_model(df_test, lr, dv, categorical, numerical, target)

    print(f"Train MSE = {mse_train}, Validation MSE = {mse_val}, Test MSE = {mse_test}")

    dump_model(lr, dv)
    print("baseline model successfully dumped")


def dump_model(lr: object, dv: object) -> None:
    """dump model and vectors"""
    with open("models/lin_reg.bin", "wb") as f_out:
        pickle.dump((dv, lr), f_out)


if __name__ == "__main__":
    price_prediction()
