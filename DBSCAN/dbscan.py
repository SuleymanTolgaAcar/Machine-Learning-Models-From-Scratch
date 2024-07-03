import numpy as np


class DBSCAN:

    def __init__(self, radius, min_neighbors) -> None:
        self.radius = radius
        self.min_neighbors = min_neighbors

    def _get_neighbors(self, X, point):
        neighbors = []
        for i in range(len(X)):
            if np.linalg.norm(X[i] - point) <= self.radius:
                neighbors.append(i)
        return np.array(neighbors)

    def _expand_cluster(self, X, labels, point, neighbors, cluster):
        labels[point] = cluster
        i = 0
        while i < len(neighbors):
            p = neighbors[i]
            if labels[p] == -1:
                labels[p] = cluster
            elif labels[p] == 0:
                labels[p] = cluster
                p_neighbors = self._get_neighbors(X, X[p])
                if len(p_neighbors) >= self.min_neighbors:
                    neighbors = np.concatenate([neighbors, p_neighbors])
            i += 1

    def predict(self, X):
        labels = np.zeros(len(X))
        cluster = 0
        for i in range(len(X)):
            if labels[i] != 0:
                continue
            neighbors = self._get_neighbors(X, X[i])
            if len(neighbors) < self.min_neighbors:
                labels[i] = -1
            else:
                cluster += 1
                self._expand_cluster(X, labels, i, neighbors, cluster)
        return labels
