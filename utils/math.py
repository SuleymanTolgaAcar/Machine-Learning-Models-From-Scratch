import numpy as np


def normalize(data):
    return (data - data.min()) / (data.max() - data.min())


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
