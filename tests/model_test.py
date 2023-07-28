from pathlib import Path

import model


def test_prepare_features():
    model_service = model.ModelService(None)

    listing = {
        "neighbourhood": "Test_neighbourhood",
        "room_type": "Test_room_type",
        "availability_365": 3.66,
        "price": 300
    }

    actual_features = model_service.prepare_features(listing)

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

    actual_features = model_service.prepare_features(listing)

    expected_fetures = {
        "neighbourhood": "Test_neighbourhood",
        "room_type": "Test_room_type",
        "availability_365": 3.66,
        "price": 0
    }

    assert actual_features == expected_fetures



class ModelMock:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        n = len(X)
        return [self.value] * n


def test_predict():
    model_mock = ModelMock(10.0)
    model_service = model.ModelService(model_mock)

    features = {
        "PU_DO": "130_205",
        "trip_distance": 3.66,
    }

    actual_prediction = model_service.predict(features)
    expected_prediction = 10.0

    assert actual_prediction == expected_prediction
