import numpy as np
from collections import defaultdict
# k-means clustering


def generate_test_points(count):
    # Generates data points to cluster
    x_coords = np.random.randint(low=0, high=100, size=count)
    y_coords = np.random.randint(low=0, high=100, size=count)
    X = np.array([[x, y] for x, y in zip(x_coords, y_coords)])
    return X


class KMeans:
    def __init__(self, k, X):
        self.k = k
        self.labels = np.zeros(len(X))
        self.data = X
        self.initialize_centroids()
        self.it_count = 0
        self.count_per_centroid = defaultdict(int)

    def initialize_centroids(self):
        # initializes centroids with random data points
        self.centroids = self.data[np.random.choice(X.shape[0], self.k, False)]

    def update_centroids(self, d_to_elt):
        # updates centroids to be the average of assigned data points
        last_centroids = self.centroids.copy()
        for idx in range(len(self.centroids)):
            self.centroids[idx] = np.mean(d_to_elt[idx], axis=0)
            self.count_per_centroid[idx] = len(d_to_elt[idx])
        if np.array_equal(last_centroids, self.centroids):
            return True
        else:
            return False

    def assign_to_centroids(self):
        # assign data points to closest centroid
        self.it_count += 1
        d = defaultdict(list)
        for idx, curr_point in enumerate(X):
            # expectation
            dist_to_centroid = [np.linalg.norm(
                curr_point - centroid) for centroid in self.centroids]
            closest = np.argmin(dist_to_centroid)
            d[closest].append(X[idx])
        return d

    def fit(self):
        # assign to and update centroids until convergence
        converged = False
        while not converged:
            d = self.assign_to_centroids()
            converged = self.update_centroids(d)
        return self.centroids


if __name__ == '__main__':
    X = generate_test_points(100)
    kmeans = KMeans(5, X)
    print(kmeans.fit())
    print(kmeans.it_count, "iterations")
    print(kmeans.count_per_centroid)
