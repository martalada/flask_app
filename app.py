#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
from math import log10

from flask import Flask
from flask import request
from flask import jsonify
import numpy as np

# Create a flask
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome"

# http://127.0.0.1:5000/predict?sl=...&pl=...
# Create an API end point
@app.route('/predict', methods=['GET'])
def get_prediction():
    # sepal length
    sepal_length = float(request.args.get('sl'))
    # sepal width
    # sepal_width = float(request.args.get('sw'))
    # petal length
    petal_length = float(request.args.get('pl'))
    # petal width
    # petal_width = float(request.args.get('pw'))

    # The features of the observation to predict
    # features = [sepal_length,
    #            sepal_width,
    #            petal_length,
    #           petal_width]

    features = [sepal_length,
                petal_length]

    print(features)
    # Load pickled model file
    with open('model.pkl', "rb") as picklefile:
        model = pickle.load(picklefile)
    print(model)
    # Predict the class using the model
    predicted_class = int(model.predict(features))

    # Return a json object containing the features and prediction
    return jsonify(features=features, predicted_class=predicted_class)


if __name__ == '__main__':
    app.run(host="0.0.0.0")


# In[ ]:




