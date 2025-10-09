import numpy as np
import pandas as pd


class KMeansClustering:
    def __init__(self, k):
        self.k = k
        self.centroids = None
        self.clusters = None

    def _assign_clusters(self, X):
        clusters = []
        for point in X:
            distances = np.sqrt(((self.centroids - point) ** 2).sum(axis=1))
            clusters.append(distances.argmin())
        return np.array(clusters)

    def _update_centroids(self, X):
        new_centroids = []
        for cluster in range(self.k):
            new_centroids.append(X[self.clusters == cluster].mean(axis=0))
        return np.array(new_centroids)

    def predict(self, X, max_iterations=1000, tolerance=1e-5):
        self.centroids = X[np.random.choice(len(X), self.k, replace=False)]

        for _ in range(max_iterations):
            self.clusters = self._assign_clusters(X)
            new_centroids = self._update_centroids(X)
            if np.max(np.abs(self.centroids - new_centroids)) < tolerance:
                break
            self.centroids = new_centroids

        self.clusters = self._assign_clusters(X)
        return self.clusters
