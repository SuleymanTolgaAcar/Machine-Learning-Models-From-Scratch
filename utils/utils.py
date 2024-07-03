import numpy as np
import pandas as pd


def normalize(data):
    return (data - data.min()) / (data.max() - data.min())


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train_test_split(X, y, test_size=0.2):
    X_train = X.iloc[: -int(len(X) * test_size)]
    y_train = y.iloc[: -int(len(y) * test_size)]
    X_test = X.iloc[-int(len(X) * test_size) :]
    y_test = y.iloc[-int(len(y) * test_size) :]
    return X_train, y_train, X_test, y_test
