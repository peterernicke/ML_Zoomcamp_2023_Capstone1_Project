import os
import numpy as np
import pandas as pd
#import seaborn as sns

#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def make_model(train_generator, learning_rate=0.001, dropout_rate=0.5):
    input_size=299
    
    base_model = keras.models.Sequential()
    classes = len(list(train_generator.class_indices.keys()))

    inputs = keras.Input(shape=(input_size, input_size, 3))
    base = base_model(inputs)

    #############################################################################################

    # Stack of convolutional layers and pooling layers
    conv_1 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu',)(base)
    vectors_1 = keras.layers.MaxPooling2D(pool_size=(2,2))(conv_1)
    conv_2 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu',)(vectors_1)
    vectors_2 = keras.layers.MaxPooling2D(pool_size=(2,2))(conv_2)
    conv_3 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu',)(vectors_2)
    vectors_3 = keras.layers.MaxPooling2D(pool_size=(2,2))(conv_3)

    # Flatten Layer
    flatten = keras.layers.Flatten()(vectors_3)

    # Fully Connected Layers
    dense_1 = keras.layers.Dense(256, activation='relu')(flatten)
    drop_1 = keras.layers.Dropout(dropout_rate)(dense_1)
    dense_2 = keras.layers.Dense(128, activation='relu')(drop_1)
    drop_2 = keras.layers.Dropout(dropout_rate)(dense_2)

    # Output Layer
    outputs = keras.layers.Dense(classes, activation='softmax')(drop_2)

    #############################################################################################

    model = keras.Model(inputs, outputs)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    return model

def get_ImageDataGenerators():

    train_gen = ImageDataGenerator(
        rescale=1./255,
        #rotation_range=30,
        #width_shift_range=10,
        #height_shift_range=10,
        #shear_range=10,
        #zoom_range=0.1,
        #cval=0.0,
        #horizontal_flip=False,
        #vertical_flip=True,
    )

    train_generator = train_gen.flow_from_directory(
        './../Data/train',
        target_size=(299, 299),
        batch_size=20,
        class_mode='categorical'
    )

    test_gen = ImageDataGenerator(rescale=1./255)

    test_generator = test_gen.flow_from_directory(
        './../Data/test',
        target_size=(299, 299),
        batch_size=20,
        shuffle=False,
        class_mode='categorical'
    )

    return train_generator, test_generator

def test_model(image_data_generator, model):
    predictions = []

    for subdir in os.listdir(image_data_generator.directory):
        subdir_path = os.path.join(image_data_generator.directory, subdir)
        
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)
                file_size = os.path.getsize(file_path)
                
                # Loading and pre processing image
                img = image.load_img(file_path, target_size=image_data_generator.image_shape)
                img_array = np.array(img, dtype="float32") / 255.0
                img_array = img_array.reshape((1,) + img_array.shape)

                # Predicting the label of that image
                prediction = model.predict(img_array)[0]

                # Saving the index with the highest probability
                predicted_class_index = np.argmax(prediction)
                predicted_class_name = list(image_data_generator.class_indices.keys())[predicted_class_index]

                # Saving the results
                result = {
                    'File Name': file,
                    'File Size': file_size,
                    'Aircraft type (Ground Truth)': subdir,
                    'Aircraft type (Prediction)': predicted_class_name,
                    **{f'Probability ({class_name})': f'{prob:.5%}' for class_name, prob in zip(image_data_generator.class_indices.keys(), prediction)}
                }
                predictions.append(result)

    # Creating Pandas DataFrame 
    predictions_df = pd.DataFrame(predictions)
    return predictions_df

def output_acc_per_class(test_image_data_generator, test_predictions):
    for aircraft_type in test_image_data_generator.class_indices:
        count = test_image_data_generator.classes.tolist().count(test_image_data_generator.class_indices[aircraft_type])
        aircraft_set = test_predictions[test_predictions['Aircraft type (Ground Truth)'] == aircraft_type]
        acc = np.sum(aircraft_set['Aircraft type (Ground Truth)'] == aircraft_set['Aircraft type (Prediction)']) / count
        print(f'{aircraft_type} Accuracy: {acc:.4f}')

checkpoint = keras.callbacks.ModelCheckpoint(
    'sequential_model_v1_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

train_generator, test_generator = get_ImageDataGenerators()

learning_rate = 0.001
droprate = 0.6

model = make_model(
    train_generator=train_generator, 
    learning_rate=learning_rate,
    dropout_rate=droprate
)

history = model.fit(
    train_generator, 
    epochs=100, 
    validation_data=test_generator,
    callbacks=[checkpoint]
)

# Evaluating the model
test_predictions = test_model(test_generator, model)

filenames = test_generator.filenames
num_samples = len(filenames)
accuracy = np.sum(test_predictions['Aircraft type (Ground Truth)'] == test_predictions['Aircraft type (Prediction)']) / num_samples
print(f'Overall Accuracy: {accuracy:.4f}')

output_acc_per_class(test_generator, test_predictions)