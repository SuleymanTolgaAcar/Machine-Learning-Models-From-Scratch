import numpy as np

from utils.math import sigmoid


class LogisticRegression:

    def __init__(self, epochs=1000, learning_rate=0.01):
        self.epochs = epochs
        self.learning_rate = learning_rate

    def _gradient_descent(self, epochs, learning_rate):
        self.b = np.zeros(self.X.shape[1])
        for _ in range(epochs):
            self.b = (
                self.b
                - learning_rate
                * (self.X.T @ (sigmoid(self.X @ self.b) - self.y))
                / self.X.shape[0]
            )

    def fit(self, X, y):
        self.X = np.c_[np.ones(X.shape[0]), X]
        self.y = y
        self._gradient_descent(self.epochs, self.learning_rate)
        return self.b

    def predict(self, X):
        y_pred = sigmoid(np.c_[np.ones(X.shape[0]), X] @ self.b)
        return (y_pred > 0.5).astype(int)
