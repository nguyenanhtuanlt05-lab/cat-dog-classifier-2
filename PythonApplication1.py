import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, k, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X):
        np.random.seed(42)
        # Initialize centroids randomly
        random_indices = np.random.permutation(X.shape[0])[: self.k]
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            self.labels = self._assign_clusters(X)
            new_centroids = self._calculate_centroids(X)

            # Check for convergence
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break

            self.centroids = new_centroids

    def predict(self, X):
        return self._assign_clusters(X)

    def _assign_clusters(self, X):
        # Calculate distances from each point to each centroid
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _calculate_centroids(self, X):
        # Calculate the mean of points in each cluster to find new centroids
        return np.array([X[self.labels == i].mean(axis=0) for i in range(self.k)])

    def inertia(self, X):
        # Calculate inertia, the sum of squared distances of samples to their
        # closest cluster center.
        return np.sum((X - self.centroids[self.labels]) ** 2)