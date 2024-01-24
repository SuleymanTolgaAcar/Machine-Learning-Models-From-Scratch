import numpy as np
import pandas as pd


class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        y_pred = []
        for point in X:
            distances = np.sqrt(((self.X - point) ** 2).sum(axis=1))
            nearest_points = self.y[distances.argsort()[: self.k]]
            y_pred.append(pd.Series(nearest_points).value_counts().index[0])
        return np.array(y_pred)
