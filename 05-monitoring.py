import os
import pickle
import logging

import s3fs
import numpy as np
import pandas as pd
import psycopg
import xgboost
from prefect import flow, task
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric

s3_bucket = os.getenv("S3_BUCKET")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)

create_table_statement = """
drop table if exists price_prediction;
create table price_prediction(
	bin integer,
	prediction_drift float,
	num_drifted_columns integer
)
"""

categorical = ["neighbourhood", "room_type"]
numerical = ["availability_365"]
target = ["price"]

column_mapping = ColumnMapping(
    prediction="prediction",
    numerical_features=numerical,
    categorical_features=categorical,
    target=None,
)

report = Report(
    metrics=[ColumnDriftMetric(column_name="prediction"), DatasetDriftMetric()]
)


def get_data():
    df = pd.read_csv(f"s3://{s3_bucket}/data/NYC-Airbnb-2023.csv", low_memory=False)
    df = df[df.price <= df.price.quantile(q=0.95)]
    df = df[categorical + numerical + target]
    df[categorical] = df[categorical].astype(str)

    reference_data = df[:20000]
    reference_data["bin"] = 1
    new_data = np.array_split(df[20000:], 10)
    for i in range(2, 12):
        new_data[i - 2]["bin"] = i

    return (reference_data, new_data)


@task(retries=2, retry_delay_seconds=5, name="prep_db")
def prep_db():
    with psycopg.connect(
        "host=localhost port=6432 user=postgres password=qwerty123", autocommit=True
    ) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='airbnb'")
        if len(res.fetchall()) == 0:
            conn.execute("create database airbnb;")
        with psycopg.connect(
            "host=localhost port=6432 dbname=airbnb user=postgres password=qwerty123"
        ) as conn:
            conn.execute(create_table_statement)


@task(retries=2, retry_delay_seconds=5, name="calculate_metrics")
def calculate_metrics(curr, reference_df, df):
    with open("models/lin_reg.bin", "rb") as f_in:
        dv, model = pickle.load(f_in)

    dicts_ref = reference_df[categorical + numerical].to_dict(orient="records")
    X_ref = dv.transform(dicts_ref)
    y_ref = reference_df[target].values
    reference = xgboost.DMatrix(X_ref, label=y_ref)
    reference_df["prediction"] = model.predict(reference)

    val_dicts = df[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)
    y_val = df[target].values
    valid = xgboost.DMatrix(X_val, label=y_val)
    df["prediction"] = model.predict(valid)

    report.run(
        reference_data=reference_df,
        current_data=df,
        column_mapping=column_mapping,
    )

    result = report.as_dict()

    bin = df["bin"].iloc[0]
    prediction_drift = result["metrics"][0]["result"]["drift_score"]
    num_drifted_columns = result["metrics"][1]["result"]["number_of_drifted_columns"]

    curr.execute(
        f"insert into price_prediction(bin, prediction_drift, num_drifted_columns) values ({bin}, {prediction_drift}, {num_drifted_columns})"
    )


@flow
def price_monitoring():
    prep_db()

    reference_data, new_data = get_data()

    with psycopg.connect(
        "host=localhost port=6432 dbname=airbnb user=postgres password=qwerty123",
        autocommit=True,
    ) as conn:
        for df in new_data:
            with conn.cursor() as curr:
                calculate_metrics(curr, reference_data, df)
            logging.info("data sent")


if __name__ == "__main__":
    price_monitoring()
