import numpy as np
import pandas as pd


class SimpleLinearRegression:
    X = None
    y = None
    b0 = None
    b1 = None

    def fit(self, X, y, gradient_descent=True, epochs=1000, learning_rate=0.001):
        self.X = X
        self.y = y
        if gradient_descent:
            self.gradient_descent(epochs, learning_rate)
        else:
            self.ordinary_least_squares()

    def gradient_descent(self, epochs, learning_rate):
        self.b0, self.b1 = 0, 0
        for _ in range(epochs):
            y_pred = self.predict(self.X)
            self.b0 -= learning_rate * (y_pred - self.y).mean()
            self.b1 -= learning_rate * ((y_pred - self.y) * self.X).mean()

    def ordinary_least_squares(self):
        X = np.array([np.ones(len(self.X)), self.X]).T
        b = np.linalg.inv(X.T @ X) @ X.T @ self.y
        self.b0, self.b1 = b[0], b[1]

    def predict(self, X):
        return self.b0 + self.b1 * X
