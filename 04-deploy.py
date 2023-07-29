import pickle

import xgboost
from flask import Flask, jsonify, request

with open("lin_reg.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)


def predict(features):
    X = dv.transform(features)
    data = xgboost.DMatrix(X)
    preds = model.predict(data)
    return float(preds[0])


app = Flask("duration-prediction")


@app.route("/04-deploy", methods=["POST"])
def predict_endpoint():
    listing = request.get_json()
    features = listing

    pred = predict(features)

    result = {"price": pred}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=9696)
