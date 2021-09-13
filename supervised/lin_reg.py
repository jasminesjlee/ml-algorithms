import numpy as np


class LinearRegression:
    def __init__(self, lr):
        self.lr = lr
        self.w = None  # (k+1, 1), including bias term

    def add_bias_to_X(self, X):
        (n, _) = X.shape
        bias_term = np.ones((n, 1))
        return np.hstack((X, bias_term))

    def initialize_w(self, k):
        self.w = np.zeros((k, 1))

    def loss(self, y, y_pred):
        # MSE
        m = y.shape[0]
        #         print(y, y_pred, (1/m) * np.sum(np.square(y_pred - y)))
        return (1 / m) * np.sum(np.square(y_pred - y))

    def gradient(self, y, y_pred, X_with_bias):
        """
        Input
        - y: (n, 1)
        - y_pred: (n, 1)
        - X_with_bias: (n, k+1)

        Output:
        - grad: (k+1, 1)
        """
        m = y.shape[0]
        return (2 / m) * (X_with_bias.T @ (y_pred - y))

    def fit(self, X, y, num_iters=10):
        # X: (n, k)
        # y: (n, 1)
        X_with_bias = self.add_bias_to_X(X)
        self.initialize_w(X_with_bias.shape[1])
        cost_list = []
        for _ in range(num_iters):
            y_pred = X_with_bias @ self.w
            cost = self.loss(y, y_pred)
            gradient = self.gradient(y, y_pred, X_with_bias)
            self.w -= self.lr * gradient
            cost_list.append(cost)
        return cost_list

    def predict(self, X, y):
        X_with_bias = self.add_bias_to_X(X)
        y_pred = X_with_bias @ self.w
        cost = self.loss(y, y_pred)
        return cost, np.round(y_pred)
