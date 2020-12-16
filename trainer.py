"""
Sentiment Analysis from Audio using Python.

Maching Learning Classifier Approach
(Using Statistical Modelling with mfcc, chroma and mel)

Made by Rakshan Sharma
"""
# Importing helper script to extract features
from feature_extract import feature_extract

# Machine Learning (ML) Imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Other Imports
import os
import glob
import joblib
from sys import argv
import numpy as np
from termcolor import cprint


# Since I'm using the RAVDESS dataset, I'm using labels as per that.
emotions = {
    '01': 'calm',
    '02': 'calm',
    '03': 'elated',
    '04': 'apprehensive',
    '05': 'fearful',
    '06': 'apprehensive',
    '07': 'angry',
    '08': 'elated'
}

# These are the emotions the model will care about
observable_emotions = ['calm', 'elated', 'angry', 'apprehensive']


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
        emotion = emotions[filename.split("-")[2]]

        # Skipping over data points not in observable_emotions
        if emotion not in observable_emotions:
            continue

        feature = feature_extract(file)

        x.append(feature)
        y.append(emotion)

    return train_test_split(np.array(x), y, test_size=test_size,
                            random_state=rs)


if __name__ == '__main__':
    try:
        int(argv[1])
    except IndexError:
        argv[1] = [0, 1]

    best_support_vector_machine_acc = 0
    best_Nearest_Neighbors_Classifier_acc = 0
    best_Gaussian_Naive_Bayes_acc = 0
    best_K_Nearest_Neighbors_Classifier_acc = 0
    best_multi_layer_perceptron_acc = 0

    for i in range(int(argv[1])):
        if int(argv[1]) == 1:
            cprint("[+] Loading Data and Extracting Features", 'green')
        else:
            cprint(f"[+] Iteration {i+1} Running...", 'green', end="\r")

        # loading data as per tree structure in dataset
        x_train, x_test, y_train, y_test = load_data(0.25)

        if int(argv[1]) == 1:
            # Printing out some relevant details
            print("Training Samples:", x_train.shape[0])
            print("Test Samples:", x_test.shape[0])
            print(f'Features extracted: {x_train.shape[1]}')
            cprint("[+] Training Classifiers", 'green')

        # Initialize ML Models
        support_vector_machine = svm.SVC()
        Nearest_Neighbors_Classifier = KNeighborsClassifier(n_neighbors=1)
        K_Nearest_Neighbors_Classifier = KNeighborsClassifier(n_neighbors=2)
        Gaussian_Naive_Bayes = GaussianNB()
        multi_layer_perceptron = MLPClassifier(
            alpha=0.01, batch_size=512, epsilon=1e-08,
            hidden_layer_sizes=(300),
            learning_rate='adaptive', max_iter=500)

        # Fit ML Models
        support_vector_machine.fit(x_train, y_train)
        Nearest_Neighbors_Classifier.fit(x_train, y_train)
        Gaussian_Naive_Bayes.fit(x_train, y_train)
        K_Nearest_Neighbors_Classifier.fit(x_train, y_train)
        multi_layer_perceptron.fit(x_train, y_train)

        # Get ML Model Predictions
        support_vector_machine_pred = support_vector_machine.predict(x_test)
        Nearest_Neighbors_Classifier_pred = Nearest_Neighbors_Classifier.predict(
            x_test)
        Gaussian_Naive_Bayes_pred = Gaussian_Naive_Bayes.predict(x_test)
        K_Nearest_Neighbors_Classifier_pred = K_Nearest_Neighbors_Classifier.predict(
            x_test)
        multi_layer_perceptron_pred = multi_layer_perceptron.predict(x_test)

        # Print ML Model out accuracy scores
        support_vector_machine_acc = accuracy_score(
            y_true=y_test, y_pred=support_vector_machine_pred)
        Nearest_Neighbors_Classifier_acc = accuracy_score(
            y_true=y_test, y_pred=Nearest_Neighbors_Classifier_pred)
        Gaussian_Naive_Bayes_acc = accuracy_score(
            y_true=y_test, y_pred=Gaussian_Naive_Bayes_pred)
        K_Nearest_Neighbors_Classifier_acc = accuracy_score(
            y_true=y_test, y_pred=K_Nearest_Neighbors_Classifier_pred)
        multi_layer_perceptron_acc = accuracy_score(
            y_true=y_test, y_pred=multi_layer_perceptron_pred)

        joblib.dump(
            support_vector_machine, f'Models_New/svm-{round(support_vector_machine_acc*100, 2)}.joblib')
        joblib.dump(
            Nearest_Neighbors_Classifier, f'Models_New/NNC-{round(Nearest_Neighbors_Classifier_acc*100, 2)}.joblib')
        joblib.dump(
            Gaussian_Naive_Bayes, f'Models_New/GNB-{round(Gaussian_Naive_Bayes_acc*100, 2)}.joblib')
        joblib.dump(
            K_Nearest_Neighbors_Classifier, f'Models_New/KNNC-{round(K_Nearest_Neighbors_Classifier_acc*100, 2)}.joblib')
        joblib.dump(
            multi_layer_perceptron, f'Models_New/MLP-{round(multi_layer_perceptron_acc*100, 2)}.joblib')

        # Keeping track of the best scores
        best_support_vector_machine_acc = max(
            support_vector_machine_acc, best_support_vector_machine_acc)
        best_Nearest_Neighbors_Classifier_acc = max(
            Nearest_Neighbors_Classifier_acc, best_Nearest_Neighbors_Classifier_acc)
        best_Gaussian_Naive_Bayes_acc = max(
            Gaussian_Naive_Bayes_acc, best_Gaussian_Naive_Bayes_acc)
        best_K_Nearest_Neighbors_Classifier_acc = max(
            K_Nearest_Neighbors_Classifier_acc, best_K_Nearest_Neighbors_Classifier_acc)
        best_multi_layer_perceptron_acc = max(
            multi_layer_perceptron_acc, best_multi_layer_perceptron_acc)

        if int(argv[1]) == 1:
            print("SVM Accuracy: {:.2f}%".format(support_vector_machine_acc*100))
            print("Nearest Neighbor Classifier Accuracy: {:.2f}%".format(
                Nearest_Neighbors_Classifier_acc*100))
            print(
                "K-Nearest Neighbor Classifier Accuracy: {:.2f}%".format(K_Nearest_Neighbors_Classifier_acc*100))
            print("Naive Bayes Classifier Accuracy: {:.2f}%".format(
                Gaussian_Naive_Bayes_acc*100))
            print("Multi Layer Perceptron Accuracy: {:.2f}%".format(
                multi_layer_perceptron_acc*100))
        elif i == (int(argv[1]) - 1):
            print("Best SVM Accuracy: {:.2f}%".format(
                best_support_vector_machine_acc*100))
            print("Best Nearest Neighbor Classifier Accuracy: {:.2f}%".format(
                best_Nearest_Neighbors_Classifier_acc*100))
            print("Best K-Nearest Neighbor Classifier Accuracy: {:.2f}%".format(
                best_K_Nearest_Neighbors_Classifier_acc*100))
            print("Best Naive Bayes Classifier Accuracy: {:.2f}%".format(best_Gaussian_Naive_Bayes_acc*100))
            print("Best Multi Layer Perceptron Accuracy: {:.2f}%".format(best_multi_layer_perceptron_acc*100))
