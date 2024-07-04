import numpy as np
import pandas as pd


class MultipleLinearRegression:

    def __init__(self, epochs=1000, learning_rate=0.01, algorithm="gradient_descent"):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.algorithm = algorithm

    def _gradient_descent(self, epochs, learning_rate):
        self.b = np.zeros(self.X.shape[1])
        for _ in range(epochs):
            self.b = (
                self.b
                - learning_rate
                * (self.X.T @ (self.X @ self.b - self.y))
                / self.X.shape[0]
            )

    def _ordinary_least_squares(self):
        self.b = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y

    def fit(self, X, y):
        self.X = np.c_[np.ones(X.shape[0]), X]
        self.y = y
        if self.algorithm == "gradient_descent":
            self._gradient_descent(self.epochs, self.learning_rate)
        elif self.algorithm == "ordinary_least_squares":
            self._ordinary_least_squares()
        return self.b

    def predict(self, X):
        X_with_ones = np.c_[np.ones(X.shape[0]), X]
        return X_with_ones @ self.b
