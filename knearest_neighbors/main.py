import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .k_nearest_neighbors import KNearestNeighbors

df = pd.read_csv("knearest_neighbors/KNNAlgorithmDataset.csv")
y = df["diagnosis"].values
X = df[["radius_mean", "texture_mean"]].values

model = KNearestNeighbors(k=11)
model.fit(X, y)

center = np.mean(X, axis=0)
spread = np.max(X, axis=0) - np.min(X, axis=0)
points = np.random.uniform(center - spread / 3, center + spread / 3, size=(5, 2))
y_pred = model.predict(points)

plt.scatter(
    X[:, 0], X[:, 1], c=pd.Series(y).map({"B": "royalblue", "M": "#fa3737"}), s=15
)
plt.scatter(
    points[:, 0],
    points[:, 1],
    c=pd.Series(y_pred).map({"B": "midnightblue", "M": "#851010"}),
    s=75,
    marker="x",
    linewidths=4,
)
plt.show()
