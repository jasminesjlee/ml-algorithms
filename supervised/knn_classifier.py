import numpy as np


class KNNClassifier:
    def __init__(self, k):
        self.k = k
        self.train_X = None
        self.train_y = None

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def euclid_dist(self, p1, p2):
        return np.sum(np.sqrt(np.square(p1 - p2)))

    def predict(self, X):
        preds = []
        for r in X:
            out = np.zeros(X.shape[0])
            for i, train_r in enumerate(self.train_X):
                out[i] = self.euclid_dist(train_r, r)
            smallest_k_idx = np.argsort(out)[: self.k]
            smallest_k_labels = self.train_y[smallest_k_idx]
            preds.append(np.argmax(np.bincount(smallest_k_labels)))
        return preds


knn_class = KNNClassifier(1)
X = np.array([[4], [4], [4], [0], [0], [0]])
y = np.array([1, 1, 1, 0, 0, 0])
knn_class.fit(X, y)

X = np.array([[4], [3], [2], [1], [0], [-1]])
y = np.array([[1], [1], [1], [-1], [-1], [-1]])
knn_class.predict(X)
