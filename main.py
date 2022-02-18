import random

import numpy as np
import pandas as pd

import mlp
from mlp import Mlp


def read_iris_dataset():
    """
    Loads the iris dataset and then labels the flowers using one hot encoding

    Returns
    -------
    X, Y - list, list
        X the train data and Y the labels of the data
    """
    dataset = pd.read_csv("iris.csv")
    X = dataset[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    X = X.values
    Y = []
    for i in range(150):
        if i < 50:
            Y.append([1., 0., 0.])
        elif i < 100:
            Y.append([0., 1., 0.])
        else:
            Y.append([0., 0., 1.])
    return X, Y


def model_performance():
    """
    Loads the iris dataset shuffles it, splits it 120/30 and then trains and returns the
    results of the model.

    Returns
    -------
    the number of correct predictions the model made on 30 test data
    """
    # load data
    X, Y = read_iris_dataset()

    # X = [1, 2, 3, 4, 5]
    # Y = [11, 22, 33, 44, 55]

    # shuffle data
    zipped = list(zip(X, Y))
    random.shuffle(zipped)
    X, Y = zip(*zipped)

    # split train and test
    X_train = X[:120]
    Y_train = Y[:120]
    X_test = X[120:]
    Y_test = Y[120:]

    # define and fit model
    model = mlp.Mlp(layer_layout=[4, 5, 3], activation_function='sigmoid', epochs=500)
    model.fit(X_train, Y_train)

    # test model
    counter = 0
    for sample, label in zip(X_test, Y_test):
        predict = np.around(model.predict(sample))
        flag = True
        for x, y in zip(predict, label):
            if x != y:
                flag = False
        counter += flag
    return counter


if __name__ == '__main__':
    """
    Test the performance of the model over 3 iterations and then prints the performance and the accuracy of the model
    """
    performance = 0
    for i in range(3):
        performance += model_performance()
    performance = performance // 3
    print(f'Average performance of the model (120/30 split) over 3 trainings is : {performance}')
    print(f'Predicted accuracy is : {(performance * 100) / 30}')
