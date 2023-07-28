import mlflow
import boto3
import os

run_id = os.getenv('RUN_ID')
print(run_id)

def prepare_features(listing):
    features = {}
    features['neighbourhood'] = listing['neighbourhood']
    features['room_type'] = listing['room_type']
    features['availability_365'] = listing['availability_365']
    features['price'] = listing['price'] if listing['price'] < 500 else 0
    return features

def get_model_location(run_id):
    model_location = os.getenv('MODEL_LOCATION')

    if model_location is not None:
        return model_location

    model_bucket = os.getenv('MODEL_BUCKET', 'mlops-course-project/mlflow')
    experiment_id = os.getenv('MLFLOW_EXPERIMENT_ID', '2')

    model_location = f's3://{model_bucket}/{experiment_id}/{run_id}/artifacts/models_mlflow'
    return model_location


def load_model(run_id):
    model_path = get_model_location(run_id)
    model = mlflow.pyfunc.load_model(model_path)
    return model



def test_prepare_features():
    listing = {
        "neighbourhood": "Test_neighbourhood",
        "room_type": "Test_room_type",
        "availability_365": 3.66,
        "price": 300
    }
    actual_features = prepare_features(listing)
    expected_fetures = {
        "neighbourhood": "Test_neighbourhood",
        "room_type": "Test_room_type",
        "availability_365": 3.66,
        "price": 300
    }
    assert actual_features == expected_fetures


    listing = {
        "neighbourhood": "Test_neighbourhood",
        "room_type": "Test_room_type",
        "availability_365": 3.66,
        "price": 600
    }
    actual_features = prepare_features(listing)
    expected_fetures = {
        "neighbourhood": "Test_neighbourhood",
        "room_type": "Test_room_type",
        "availability_365": 3.66,
        "price": 0
    }
    assert actual_features == expected_fetures


def test_predict():
    model = load_model(run_id)

    
    
    model_mock = ModelMock(10.0)
    model_service = model.predict(model_mock)

    features = {
        "PU_DO": "130_205",
        "trip_distance": 3.66,
    }

    actual_prediction = model_service.predict(features)
    expected_prediction = 10.0

    assert actual_prediction == expected_prediction
