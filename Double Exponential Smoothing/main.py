import numpy as np
import matplotlib.pyplot as plt


def MAPE(y, y_pred):
    return round(100 * np.nanmean(np.abs(y - y_pred) / np.abs(y)), 2)


def double_exponential_smoothing(x, l_zero, b_zero=0, alpha=0.3, beta=0.5, mape=False):
    if alpha < 0 or alpha > 1:
        raise ValueError("Invalid alpha")
    if beta < 0 or beta > 1:
        raise ValueError("Invalid beta")
    y = np.array(x)
    l = np.zeros_like(y)
    b = np.zeros_like(y)
    l[0] = l_zero
    b[0] = b_zero

    for t in range(1, len(y)):
        l[t] = alpha * y[t - 1] + (1 - alpha) * (l[t - 1] + b[t - 1])
        b[t] = beta * (l[t] - l[t - 1]) + (1 - beta) * b[t - 1]

    l[0] = np.nan
    b[0] = np.nan

    if mape:
        return l, MAPE(y, l)
    else:
        return l


x = np.array([2.92, 0.84, 2.69, 2.42, 1.83, 1.22, 0.10, 1.32, 0.56, -0.35])
l_values = double_exponential_smoothing(x, l_zero=0, alpha=0.8, beta=0.2)
plt.plot(x, label="Original")
plt.plot(l_values, label="Smoothed")
plt.legend()
plt.show()
