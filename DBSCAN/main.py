import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dbscan import DBSCAN

blob1 = np.random.randn(30, 2) + np.array([0, 2.2])
blob2 = np.random.randn(30, 2) + np.array([-2.2, -2.2])
blob3 = np.random.randn(30, 2) + np.array([2.2, -2.2])
X = np.concatenate((blob1, blob2, blob3))

model = DBSCAN(radius=1, min_neighbors=5)

labels = model.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.show()
