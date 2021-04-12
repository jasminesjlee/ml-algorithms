import numpy as np


class SVM():
    def __init__(self, lr, num_iters):
        self.w = None  # w will have shape (p+1, 1)
        self.lr = lr
        self.num_iters = num_iters

    def loss(self, W, X, Y):
        # W: (p, 1)
        # X : (n, p)
        # Y = (n, 1)

        num_samples = Y.shape[0]
        # (n,1) * ((n, p+1) * (p+1,1)) --> (n,1)
        distances = 1 - Y.reshape((num_samples, 1)) * np.dot(X, W)
        distances = distances[distances > 0]  # (n,1)
        hinge_loss = sum(distances) / num_samples  # (n,1)
        cost = 1 / 2 * np.dot(W.T, W) + hinge_loss  # int
        return cost

    def loss_deriv(self, W, X, Y):
        # (n,1) * ((n, p+1) * (p+1,1)) --> (n,1)
        n = Y.shape[0]
        distances = (1 - Y.reshape((n, 1)) * np.dot(X, W)).reshape(n)
        # X, Y = X[distances > 0, :], Y[distances > 0]
        # # (1, n) X (n,p) --> (1,p)
        # gradients = np.sum(-Y.reshape((6, 1)) * X, axis=0)
        # return 1/n * gradients
        dw = np.zeros(W.shape)
        for ind, d in enumerate(distances):
            if max(0, d) == 0:
                di = W
            else:
                di = W - ((Y[ind] * X[ind])).reshape(W.shape)
            dw += di
        dw = dw/len(Y)  # average
        return dw

    def sgd_fit(self, X, y):
        (num_samples, num_feats) = X.shape  # (p, n)
        self.w = np.zeros((num_feats + 1, 1))  # (p+1, 1)
        X = np.hstack((X, np.ones((num_samples, 1))))
        for _ in range(self.num_iters):
            error_deriv = self.loss_deriv(self.w, X, y)
            self.w = self.w - self.lr * \
                error_deriv.reshape((self.w.shape[0], 1))
            print(self.loss(self.w, X, y))

    def predict(self, X, y):
        (num_samples, num_feats) = X.shape  # (p, n)
        total_error = 0.0
        X = np.hstack((X, np.ones((num_samples, 1))))
        print(np.dot(X, self.w))
        y_pred = np.sign(np.dot(X, self.w))
        total_error = self.loss(self.w, X, y)

        return total_error, y_pred


if __name__ == '__main__':
    lr = SVM(0.01, 5000)
    X = np.array([[1], [2], [3], [100], [500], [200]])  # (n, p)
    y = np.array([-1, -1, -1, 1, 1, 1])  # (n,)
    lr.sgd_fit(X, y)
    print(lr.predict(X, y))
