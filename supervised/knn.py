import numpy as np


class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

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
        for v1, true_y in zip(X, Y):
            distances = np.zeros(
                self.X.shape[0],
            )
            for idx, v2 in enumerate(self.X):
                distances[idx] = self.euclid_dist(v1, v2)
            closest_point_idx = np.argsort(
                distances
            )  # idx of closest point to furthest point
            bin_count = np.bincount(self.Y.astype(int)[closest_point_idx[: self.k]])
            y_pred.append((np.argmax(bin_count), true_y))
        return y_pred


dataset = np.array(
    [
        [2.7810836, 2.550537003, 0],
        [1.465489372, 2.362125076, 0],
        [3.396561688, 4.400293529, 0],
        [1.38807019, 1.850220317, 0],
        [3.06407232, 3.005305973, 0],
        [7.627531214, 2.759262235, 1],
        [5.332441248, 2.088626775, 1],
        [6.922596716, 1.77106367, 1],
        [8.675418651, -0.242068655, 1],
        [7.673756466, 3.508563011, 1],
    ]
)
X, Y = dataset[:, :-1], dataset[:, -1]
knn = KNN(3)
knn.fit(X, Y)
print(knn.classification_predict(X, Y))