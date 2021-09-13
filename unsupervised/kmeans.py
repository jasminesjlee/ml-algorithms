import numpy as np


class KMeans:
    def __init__(self, k):
        self.k = k
        self.centroids = None
        self.centroids_to_point = [[] for _ in range(self.k)]

    def init_centroids(self, X):
        self.centroids = X[np.random.choice(X.shape[0], size=self.k, replace=False)]

    def cluster(self, X):
        self.init_centroids(X)
        prev_centroids = None
        i = 0
        while prev_centroids is None or not (self.centroids == prev_centroids).all():
            i += 1
            centroid_sum = np.zeros((self.k, X[0].shape[0]))
            centroid_total = [0 for _ in range(self.k)]
            self.centroids_to_point = [[] for _ in range(self.k)]
            prev_centroids = self.centroids
            for p in X:
                centroid_idx = np.argmin(
                    np.array([np.linalg.norm(p - c) for c in self.centroids])
                )
                centroid_sum[centroid_idx] += p
                centroid_total[centroid_idx] += 1
                self.centroids_to_point[centroid_idx].append(p.copy())
            # re evaluate centroids
            for i, (c_sum, c_count) in enumerate(zip(centroid_sum, centroid_total)):
                self.centroids[i] = c_sum / c_count
        return self.centroids


if __name__ == "__main__":
    km = KMeans(5)
    X = np.array([[0, 0], [0, 1], [0, 2], [1, 2], [1, 3], [1, 4]])
    print(km.cluster(X))
