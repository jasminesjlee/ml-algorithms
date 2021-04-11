import numpy as np


class LinearRegression():
    def __init__(self, lr, num_iters):
        self.w = None  # w will have shape (p+1, 1)
        self.lr = lr
        self.num_iters = num_iters

    def sgd_fit(self, X, y):
        (num_feats, num_samples) = X.shape  # (p, n)
        self.w = np.zeros(num_feats + 1)  # (p+1, 1)
        X = np.vstack((X, np.ones((1, num_samples))))  # (p+1, n)
        for _ in range(self.num_iters):
            for idx in range(num_samples):
                x = X[:, idx]
                y_pred = np.dot(x, self.w)
                error_deriv = (2/num_samples) * (y_pred - y[idx]) * x
                self.w = self.w - self.lr * error_deriv

    def gd_fit(self, X, y):
        (p, num_samples) = X.shape  # (p, n)
        self.w = np.zeros(p + 1)  # (p+1, 1)
        X = np.vstack((X, np.ones((1, num_samples))))  # (p+1, n)
        # (n, p+1) * (p+1, 1) --> (n, 1)
        for _ in range(self.num_iters):
            y_pred = np.dot(X.T, self.w.reshape((p+1, 1)))
            error_deriv = np.sum(((2/num_samples) * np.sum((y_pred.T - y.reshape((1, num_samples))))
                                  * X), axis=1)/num_samples  # (1, n) * (n * p+1) --> (1, p+1)
            self.w = self.w - self.lr * error_deriv

    def predict(self, X, y):
        (num_feats, num_samples) = X.shape  # (p, n)
        total_error = 0.0
        X = np.vstack((X, np.ones((1, num_samples))))  # (p+1, n)
        y_pred_list = []
        for idx in range(num_samples):
            x = X[:, idx]
            y_pred = np.dot(x, self.w)
            total_error += -(1/num_samples) * np.sum((y[idx] - y_pred)**2)
            y_pred_list.append(y_pred)
        return total_error, y_pred_list


if __name__ == '__main__':
    lr = LinearRegression(0.001, 100)
    X = np.array([[1, 2, 3, 4, 5, 6]])
    y = np.array([1, 2, 3, 4, 5, 6])
    lr.gd_fit(X, y)
    print(lr.predict(X, y))
