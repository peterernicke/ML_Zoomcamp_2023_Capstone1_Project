#!/usr/bin/env python
# coding: utf-8

import requests 
import numpy as np
import tflite_runtime.interpreter as tflite

from PIL import Image

model_file = 'final-model.tflite'
classes = ['A400M', 'C130', 'Su57', 'Tu160']
#url = 'https://cdn.pixabay.com/photo/2021/01/13/14/51/airbus-a400m-atlas-5914332_960_720.jpg'

def get_interpreter():
    interpreter = tflite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    return interpreter, input_index, output_index
 
def predict(img):
    # Preprocessing the image
    x = np.array(img)
    x = np.float32(x) / 255.
    # Turning this image into a batch of one image
    X = np.array([x])

    interpreter, input_index, output_index = get_interpreter()

    # Initializing the input of the interpreter with the image X
    interpreter.set_tensor(input_index, X)

    # Invoking the computations in the neural network
    interpreter.invoke()

    # Results are in the output_index. so fetching the results...
    preds = interpreter.get_tensor(output_index)

    #predicted_class_index = np.argmax(preds)
    #predicted_class_name = classes[predicted_class_index]
    #score = dict(zip(classes, preds[0]))
 
    # What happens here is we take an Numpy array and 
    # it will be converted to usual python list with usual python floats.
    float_predictions = preds[0].tolist()
 
    return dict(zip(classes, float_predictions))
 
def lambda_handler(event, context):
    url = event['url']
    
    response = requests.get(url)
    if response.status_code == 200:
        with open("aircraft.jpg", 'wb') as f:
            f.write(response.content)

    with Image.open('./aircraft.jpg') as img:
        img = img.resize((299,299), Image.NEAREST)

    result = predict(img)
    return result