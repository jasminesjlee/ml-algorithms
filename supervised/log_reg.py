import numpy as np


class LogisticRegression:
    def __init__(self, lr, sgd=False):
        self.lr = lr
        self.sgd = sgd
        self.loss_vals = []
        self.w = None

    def loss(self, y, y_pred):
        eps = 1e-5  # to avoid log(0) error
        return (-1 / y_pred.shape[0]) * np.sum(
            (y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))
        )

    def grad(self, y, y_pred, X):
        return (X.T @ (y_pred - y)) / X.shape[0]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y, num_iters=10):
        # X shape: (n, k)
        # y shape: (n, 1)
        # w shape: (k, 1)
        (n, k) = X.shape
        self.w = np.zeros((k + 1, 1))

        for _ in range(num_iters):
            if self.sgd:
                self.sgd_fit(X, y)
            else:
                self.batchgd_fit(X, y)

    def add_bias_to_feats(self, X):
        (n, k) = X.shape
        X = np.hstack((X, np.ones((n, 1))))
        return X

    def batchgd_fit(self, X, y):
        X = self.add_bias_to_feats(X)
        y_pred = self.sigmoid(X @ self.w)
        y = y.reshape((y.shape[0], 1))
        grad = self.grad(y, y_pred, X)
        self.w -= self.lr * grad
        self.loss_vals.append(self.loss(y_pred, y))

    def sgd_fit(self, X, y):
        loss = 0
        X = self.add_bias_to_feats(X)
        for x, label in zip(X, y):
            y_pred = self.sigmoid(x @ self.w)
            label = np.array([label])
            grad = self.grad(label, y_pred, x.reshape((1, x.shape[0])))
            grad = grad.reshape((x.shape[0], 1))
            self.w -= self.lr * grad
            loss += self.loss(y_pred, label)
        self.loss_vals.append(loss / X.shape[0])

    def predict(self, X, y):
        num_samples = X.shape[0]
        X = self.add_bias_to_feats(X)
        y_pred = self.sigmoid(X @ self.w)
        loss = self.loss(y, y_pred)
        return loss / num_samples, np.round(y_pred)


if __name__ == "__main__":
    lr = LogisticRegression(0.001, sgd=True)
    X = np.array([[4], [4], [4], [0], [0], [0]])
    y = np.array([1, 1, 1, 0, 0, 0])
    lr.fit(X, y, num_iters=1)
    print(lr.predict(X, y))
    print(lr.w)
