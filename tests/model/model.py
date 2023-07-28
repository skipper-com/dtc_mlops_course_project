import os
import json

import boto3
import mlflow


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


class ModelService:
    def __init__(self, model, model_version=None, callbacks=None):
        self.model = model
        self.model_version = model_version
        self.callbacks = callbacks or []

    def prepare_features(self, listing):
        features = {}
        features['neighbourhood'] = listing['neighbourhood']
        features['room_type'] = listing['room_type']
        features['availability_365'] = listing['availability_365']
        features['price'] = listing['price'] if listing['price'] < 500 else 0

        return features

    def predict(self, features):
        pred = self.model.predict(features)
        return float(pred[0])