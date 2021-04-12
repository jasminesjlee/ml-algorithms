import numpy as np


class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, Y):
        self.X = X
        self.y = y

    def euclid_dist(self, v1, v2):
        return np.sqrt(np.sum((v1 - v2) ** 2))

    def regression_predict(self, X, Y):
        assert X.shape[1] == self.X.shape[1]
        y_pred = []
        for v1 in X:
            distances = np.zeros(
                self.X.shape[0],
            )
            for idx, v2 in enumerate(self.X):
                distances[idx] = euclid_dist(v1, v2)
            closest_point_idx = np.argsort(
                distances
            )  # idx of closest point to furthest point
            y_pred.append(np.mean(Y[closest_point_idx[: self.k]]))
        return y_pred

    def classification_predict(self, X, Y):
        assert X.shape[1] == self.X.shape[1]
        y_pred = []
        for v1 in X:
            distances = np.zeros(
                self.X.shape[0],
            )
            for idx, v2 in enumerate(self.X):
                distances[idx] = euclid_dist(v1, v2)
            closest_point_idx = np.argsort(
                distances
            )  # idx of closest point to furthest point
            bin_count = np.bincount(Y[closest_point_idx[:k]])
            y_pred.append(np.argmax(bin_count))
        return y_pred
