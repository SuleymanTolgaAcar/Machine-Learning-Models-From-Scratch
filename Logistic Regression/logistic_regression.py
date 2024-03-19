import numpy as np
import pandas as pd


class LogisticRegression:
    X = None
    y = None
    b = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, epochs=1000, learning_rate=0.01):
        self.X = np.c_[np.ones(X.shape[0]), X]
        self.y = y
        self.gradient_descent(epochs, learning_rate)
        return self.b

    def gradient_descent(self, epochs, learning_rate):
        self.b = np.zeros(self.X.shape[1])
        for _ in range(epochs):
            self.b = (
                self.b
                - learning_rate
                * (self.X.T @ (self.sigmoid(self.X @ self.b) - self.y))
                / self.X.shape[0]
            )

    def predict(self, X):
        y_pred = self.sigmoid(np.c_[np.ones(X.shape[0]), X] @ self.b)
        return (y_pred > 0.5).astype(int)
