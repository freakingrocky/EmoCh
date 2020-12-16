"""Neural network train file."""
import os
import joblib
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from termcolor import cprint

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from feature_extract import feature_extract

from trainer import load_data

EPOCHS = 0

# Since I'm using the RAVDESS dataset, I'm using labels as per that.
emotions = {
    1: 'calm',
    2: 'calm',
    3: 'elated',
    4: 'apprehensive',
    5: 'fearful',
    6: 'apprehensive',
    7: 'angry',
    8:  'elated'
}

mapping = {
    '01': 1,
    '02': 1,
    '03': 2,
    '04': 3,
    '05': 0,
    '06': 3,
    '07': 4,
    '08': 2
}

# These are the emotions the model will care about
observable_emotions = [1, 2, 3, 4]


def load_data(test_size: float, rs=None):
    """
    Load amd extract features from data and return test and train sets.

    Parameters:
        - test_size: Percentage of data to use as testing set (0-1)
        - rs: Random Seed (For reproducibility) [Integer]

    """
    # Creating empty lists to populate later on
    x, y = [], []

    for file in glob.glob("./Data/Actor_*/*.wav"):
        filename = os.path.basename(file)
        emotion = mapping[(filename.split("-")[2])]

        # Skipping over data points not in observable_emotions
        if emotion not in observable_emotions:
            continue

        feature = feature_extract(file)

        x.append(feature)
        y.append(emotion)

    return train_test_split(np.array(x), np.array(y), test_size=test_size,
                            random_state=rs)


def get_model(XShape, YShape):
    model = tf.keras.models.Sequential([
        # Convolutional Layer with 43 fiters, 3*3 kernel, relu activation
        tf.keras.layers.Conv2D(
            43, (3, 3), activation="relu",
            input_shape=(XShape, YShape, 1),
            padding='same'
        ),

        # Convolutional Layer with 43 fiters, 3*3 kernel, relu activation
        tf.keras.layers.Conv2D(
            43, (3, 3), activation="relu",
            input_shape=(XShape, YShape, 1)
        ),

        # Flatten units
        tf.keras.layers.Flatten(),
        # Add 2 hidden layers with dropouts in each layer
        tf.keras.layers.Dense(200, activation="relu"),
        tf.keras.layers.Dense(200, activation="relu"),

        # Final layer
        tf.keras.layers.Dense(len(observable_emotions), activation="softmax")
    ])

    # Compiling the neural network
    compiled_model = model.compile(loss='sparse_categorical_crossentropy',
                                   optimizer='rmsprop',
                                   metrics=['accuracy'])

    return compiled_model


def train_neural_network():
    """Train the neural network."""
    cprint("[*] Data Loading", 'yellow')
    X_train, X_test, y_train, y_test = load_data(0.25)

    cprint("[+] Data Loaded", 'green')
    print(X_train.shape, X_test.shape)

    cprint("[*] Compiling Model", 'yellow')
    model = get_model(X_train.shape, y_train.shape)
    cprint("[+] Model Compiled", 'green')
    cprint("[*] Training...", 'yellow')
    model.fit(X_train, y_train,
              batch_size=16, epochs=EPOCHS,
              validation_data=(X_test, y_test))
    cprint("[+] Training Complete.", 'green')

    predictions = model.predict_classes(X_test)
    new_y_test = y_test.astype(int)
    matrix = confusion_matrix(new_y_test, predictions)

    print(classification_report(new_y_test, predictions))
    print(matrix)

    model_name = 'nn.h5'

    # Save model and weights
    if not os.path.isdir("./Model/nn.keras"):
        os.makedirs("./Model/nn.keras")
    model_path = os.path.join("./Model/nn.keras", model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)


if __name__ == '__main__':
    train_neural_network()
