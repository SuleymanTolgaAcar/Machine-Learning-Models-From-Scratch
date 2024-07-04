import numpy as np
import matplotlib.pyplot as plt

from utils.metrics import MAPE
from .double_exponential_smoothing import DoubleExponentialSmoothing


X = np.array([2.92, 0.84, 2.69, 2.42, 1.83, 1.22, 0.10, 1.32, 0.56, -0.35])

model = DoubleExponentialSmoothing(alpha=0.8, beta=0.2)
l_values = model.predict(X)

mape = MAPE(X, l_values)
print(f"MAPE: {mape}")


plt.plot(X, label="Original")
plt.plot(l_values, label="Smoothed")
plt.legend()
plt.show()
