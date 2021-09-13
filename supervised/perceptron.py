import numpy as np


class Perceptron:
    def __init__(self, lr):
        self.lr = lr
        self.w = None

    def fit(self, X, y, num_iters=10):
        # X: (n, k)
        # y: (n, 1)
        # w = (k+1, 1)

        (n, k) = X.shape
        self.w = np.zeros((k + 1, 1))
        X_with_bias = np.hstack((X, np.ones((n, 1))))
        for _ in range(num_iters):
            incorrect = 0
            for x_i, y_i in zip(X_with_bias, y):
                y_pred = (x_i @ self.w)[0]
                if y_i * y_pred <= 0:
                    incorrect += 1
                    self.w += self.lr * y_i * x_i.T.reshape(self.w.shape)

    def predict(self, X, y):
        (n, k) = X.shape
        X_with_bias = np.hstack((X, np.ones((n, 1))))
        y_preds = X_with_bias @ self.w
        out = []
        for y_pred in y_preds:
            if y_pred <= 0:
                out.append(-1)
            else:
                out.append(1)
        return out


p = Perceptron(0.1)
X = np.array([[4], [4], [4], [0], [0], [0]])
y = np.array([[1], [1], [1], [-1], [-1], [-1]])
p.fit(X, y, num_iters=10)
p.predict(X, y)
