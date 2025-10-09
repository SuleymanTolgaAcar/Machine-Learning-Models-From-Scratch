import numpy as np
import pandas as pd


class SimpleLinearRegression:

    def __init__(self, epochs=1000, learning_rate=0.001, algorithm="gradient_descent"):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.algorithm = algorithm

    def _gradient_descent(self, epochs, learning_rate):
        self.b0, self.b1 = 0, 0
        for _ in range(epochs):
            y_pred = self.predict(self.X)
            self.b0 -= learning_rate * (y_pred - self.y).mean()
            self.b1 -= learning_rate * ((y_pred - self.y) * self.X).mean()

    def _ordinary_least_squares(self):
        X = np.array([np.ones(len(self.X)), self.X]).T
        b = np.linalg.inv(X.T @ X) @ X.T @ self.y
        self.b0, self.b1 = b[0], b[1]

    def fit(self, X, y):
        self.X = X
        self.y = y
        if self.algorithm == "gradient_descent":
            self._gradient_descent(self.epochs, self.learning_rate)
        elif self.algorithm == "ordinary_least_squares":
            self._ordinary_least_squares()

    def predict(self, X):
        return self.b0 + self.b1 * X
