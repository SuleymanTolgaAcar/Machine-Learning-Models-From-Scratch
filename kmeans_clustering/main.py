import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .k_means_clustering import KMeansClustering

blob1 = np.random.randn(30, 2) + np.array([0, 2.2])
blob2 = np.random.randn(30, 2) + np.array([-2.2, -2.2])
blob3 = np.random.randn(30, 2) + np.array([2.2, -2.2])
X = np.concatenate((blob1, blob2, blob3))

k = 3
model = KMeansClustering(k=k)
clusters = model.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=clusters)
plt.scatter(
    model.centroids[:, 0],
    model.centroids[:, 1],
    c=np.arange(k),
    marker="x",
    s=100,
)
plt.show()
