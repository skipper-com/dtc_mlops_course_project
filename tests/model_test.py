import os
import pickle

import boto3
import xgboost


def prepare_features(listing):
    features = {}
    features["neighbourhood"] = listing["neighbourhood"]
    features["room_type"] = listing["room_type"]
    features["availability_365"] = listing["availability_365"]
    features["price"] = listing["price"] if listing["price"] < 500 else 0
    return features


def load_model():
    model_bucket = os.getenv("MODEL_BUCKET", "mlops-course-project")
    experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID", "2")
    run_id = os.getenv("RUN_ID", "b0f1202cf1da42439539dbf947081ac3")

    file = f"mlflow/{experiment_id}/{run_id}/artifacts/models_pickle/lin_reg.bin"

    cred = boto3.Session().get_credentials()
    ACCESS_KEY = cred.access_key
    SECRET_KEY = cred.secret_key

    s3client = boto3.client(
        "s3",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )

    response = s3client.get_object(Bucket=model_bucket, Key=file)

    body = response["Body"].read()
    dv, model = pickle.loads(body)
    return (dv, model)


def test_prepare_features():
    listing = {
        "neighbourhood": "Test_neighbourhood",
        "room_type": "Test_room_type",
        "availability_365": 3.66,
        "price": 300,
    }
    actual_features = prepare_features(listing)
    expected_fetures = {
        "neighbourhood": "Test_neighbourhood",
        "room_type": "Test_room_type",
        "availability_365": 3.66,
        "price": 300,
    }
    assert actual_features == expected_fetures

    listing = {
        "neighbourhood": "Test_neighbourhood",
        "room_type": "Test_room_type",
        "availability_365": 3.66,
        "price": 600,
    }
    actual_features = prepare_features(listing)
    expected_fetures = {
        "neighbourhood": "Test_neighbourhood",
        "room_type": "Test_room_type",
        "availability_365": 3.66,
        "price": 0,
    }
    assert actual_features == expected_fetures


def test_predict():
    dv, model = load_model()

    features = {
        "neighbourhood": "Midtown",
        "room_type": "Private room",
        "availability_365": 100,
    }

    X = dv.transform(features)
    data = xgboost.DMatrix(X)

    actual_prediction = model.predict(data)
    expected_prediction = 225.67732

    assert actual_prediction == expected_prediction
