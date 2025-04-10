import joblib
import pandas as pd
import sys
import json
import numpy as np
from flask import Flask, jsonify, request
import logging

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def call_home():
    print(request.values)
    return "SERVER IS RUNNING!"

def init():
    """
    This function is called when the container is initialized/started, usually after deployment creation/update.
    You can write logic here to perform initialization operations, such as caching the model in memory.
    """
    global model
    # desserializa o arquivo do modelo de volta em um modelo sklearn
    model = joblib.load('model.pkl')
    logging.info("Init complete")

@app.route("/score", methods=['POST'])
def run():
    """
    This function is called for every call to perform the actual prediction.
    In the example, we extract the data from the json input and call predict() from the scikit-learn model
    And we return the result in list format.
    """
    try:
        logging.info("model 1: request received")
        raw_data = request.data
        data = json.loads(raw_data)["data"]
        data = np.array(data)
        result = model.predict(data)
        logging.info("Request processed")

        return jsonify(result.tolist())
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    init()
    app.run(port=8080, host='0.0.0.0')
