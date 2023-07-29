# pylint: disable=duplicate-code
import json

import requests
from deepdiff import DeepDiff

listing = {
    "neighbourhood": "Midtown",
    "room_type": "Private room",
    "availability_365": 100,
}

url = "http://localhost:9696/04-deploy"
resp = requests.post(url, json=listing).json()
actual_prediction = resp["price"]
print("actual prediction:", actual_prediction)
expected_prediction = 225.67732

diff = DeepDiff(actual_prediction, expected_prediction, significant_digits=5)
print(f"diff={diff}")

assert "type_changes" not in diff
assert "values_changed" not in diff
