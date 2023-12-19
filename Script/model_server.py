import os
import grpc
import requests
import numpy as np
import tensorflow as tf

from PIL import Image
from flask import Flask, request, jsonify

from tensorflow import keras
from keras.preprocessing.image import load_img
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

# Creating the connection gRPC stub
host = os.getenv('TF_SERVING_HOST', 'localhost:8500')
channel = grpc.insecure_channel(host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

classes = ['A400M', 'C130', 'Su57', 'Tu160']

def np_to_protobuf(data):
    return tf.make_tensor_proto(data, shape=data.shape)

def make_request(X):
    # Using np_to_protobuf function to prepare a gRPC request
    pb_request = predict_pb2.PredictRequest()

    pb_request.model_spec.name = 'aircraft-model'
    pb_request.model_spec.signature_name = 'serving_default'
    pb_request.inputs['input_1'].CopyFrom(np_to_protobuf(X))

    return pb_request

def process_response(pb_result):
    pred = pb_result.outputs['dense_2'].float_val
    result = {c: p for c, p in zip(classes, pred)}

    return result

def preprocess_input(x):
    return x/255.

def apply_model(url):
    response = requests.get(url)
    if response.status_code == 200:
        with open("aircraft.jpg", 'wb') as f:
            f.write(response.content)

    with Image.open('./aircraft.jpg') as img:
        img = img.resize((299,299), Image.NEAREST)

    # Preprocessing the image
    #x = np.array(img, dtype='float32')

    # Turning this image into a batch of one image
    #X = np.array([x])
    #X = preprocess_input(X)

    # Preprocessing the image
    x = np.array(img)
    x = np.float32(x) / 255.
    # Turning this image into a batch of one image
    X = np.array([x])

    pb_request = make_request(X)
    pb_result = stub.Predict(pb_request, timeout=20.0)

    return process_response(pb_result)

app = Flask('Military Aircraft Detection')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.get_json()
    result = apply_model(url['url'])
    return jsonify(result)

if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)