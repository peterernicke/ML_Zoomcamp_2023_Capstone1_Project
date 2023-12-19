import tensorflow as tf
from tensorflow import keras

def convertFromH5ToTflite(h5model):
    model = keras.models.load_model(h5model)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Saving the model in tflite format
    with open('./../Model/final-model.tflite', 'wb') as f_out:
        f_out.write(tflite_model)

def convertFromH5ToSavedModelFormat(h5model):
    model = keras.models.load_model(h5model)

    # Saving the model in saved_model format
    tf.saved_model.save(model, './../Model/aircraft-model')

if __name__=="__main__":
    h5model = "./../Script/sequential_model_v1_72_0.725.h5"
    convertFromH5ToTflite(h5model)
    convertFromH5ToSavedModelFormat(h5model)