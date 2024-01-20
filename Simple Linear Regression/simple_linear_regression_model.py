import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class SimpleLinearRegressionModel:
    X = None
    y = None
    b0 = None
    b1 = None

    def fit(self, X, y, gradient_descent=True, epochs=1000, learning_rate=0.001):
        self.X = np.array([np.ones(len(X)), X]).T
        self.y = y
        if gradient_descent:
            self.gradient_descent(X, y, epochs, learning_rate)
        else:
            self.ordinary_least_squares(X, y)

    def gradient_descent(self, X, y, epochs, learning_rate):
        self.b0, self.b1 = 0, 0
        for _ in range(epochs):
            y_pred = self.predict(X)
            self.b0 -= learning_rate * (y_pred - y).mean()
            self.b1 -= learning_rate * ((y_pred - y) * X).mean()

    def ordinary_least_squares(self, X, y):
        b = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.y)
        self.b0, self.b1 = b[0], b[1]

    def predict(self, X):
        return self.b0 + self.b1 * X
