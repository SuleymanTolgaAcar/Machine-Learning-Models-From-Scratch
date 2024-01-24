import numpy as np
import pandas as pd


class MultipleLinearRegression:
    X = None
    y = None
    b = None

    def fit(self, X, y, gradient_descent=True, epochs=1000, learning_rate=0.01):
        self.X = np.c_[np.ones(X.shape[0]), X]
        self.y = y
        if gradient_descent:
            self.gradient_descent(epochs, learning_rate)
        else:
            self.ordinary_least_squares()
        return self.b

    def gradient_descent(self, epochs, learning_rate):
        self.b = np.zeros(self.X.shape[1])
        for _ in range(epochs):
            self.b = (
                self.b
                - learning_rate
                * (self.X.T @ (self.X @ self.b - self.y))
                / self.X.shape[0]
            )

    def ordinary_least_squares(self):
        self.b = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y

    def predict(self, X):
        X_with_ones = np.c_[np.ones(X.shape[0]), X]
        return X_with_ones @ self.b
