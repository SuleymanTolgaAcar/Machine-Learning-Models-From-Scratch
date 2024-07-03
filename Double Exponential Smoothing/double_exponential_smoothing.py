import numpy as np


class DoubleExponentialSmoothing:

    def __init__(self, alpha=0.3, beta=0.5, l_zero=0, b_zero=0):
        self.alpha = alpha
        self.beta = beta
        self.l_zero = l_zero
        self.b_zero = b_zero

    def predict(self, X):
        if self.alpha < 0 or self.alpha > 1:
            raise ValueError("Invalid alpha")
        if self.beta < 0 or self.beta > 1:
            raise ValueError("Invalid beta")

        y = np.array(X)
        l = np.zeros_like(y)
        b = np.zeros_like(y)
        l[0] = self.l_zero
        b[0] = self.b_zero

        for t in range(1, len(y)):
            l[t] = self.alpha * y[t - 1] + (1 - self.alpha) * (l[t - 1] + b[t - 1])
            b[t] = self.beta * (l[t] - l[t - 1]) + (1 - self.beta) * b[t - 1]

        l[0] = np.nan
        b[0] = np.nan

        return l
