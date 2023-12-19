import requests
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import tflite_runtime.interpreter as tflite

def preprocess_input(x):
    return x/255.

def get_interpreter():
    interpreter = tflite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    return interpreter, input_index, output_index

model_file = './../Model/final-model.tflite'
classes = ['A400M', 'C130', 'Su57', 'Tu160']

app = Flask('Military Aircraft Detection')

@app.route('/predict', methods=['POST'])
def predict():
    aircraft_url = request.get_json()['url']
    print(aircraft_url)

    response = requests.get(aircraft_url)
    if response.status_code == 200:
        with open("aircraft.jpg", 'wb') as f:
            f.write(response.content)

    with Image.open('./aircraft.jpg') as img:
        img = img.resize((299,299), Image.NEAREST)

    # Preprocessing the image
    x = np.array(img, dtype='float32')

    # Turning this image into a batch of one image
    X = np.array([x])
    X = preprocess_input(X)

    interpreter, input_index, output_index = get_interpreter()

    # Initializing the input of the interpreter with the image X
    interpreter.set_tensor(input_index, X)

    # Invoking the computations in the neural network
    interpreter.invoke()

    # Results are in the output_index. so fetching the results...
    preds = interpreter.get_tensor(output_index)

    predicted_class_index = np.argmax(preds)
    predicted_class_name = classes[predicted_class_index]
    #print(predicted_class_name)

    score = dict(zip(classes, preds[0]))

    #y_pred = model.predict_proba(X)[:,1]
    #score = int(y_pred[0] >= 0.50)
    #return jsonify(score)
    return(jsonify(predicted_class_name))

if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port=9797)