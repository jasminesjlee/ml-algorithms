import numpy as np


class Perceptron:
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, Y):
        # randomize weights
        # X: (n, p)
        # Y: (n, 1)
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        self.w = np.random.random(size=(X.shape[1], 1))

        for _ in range(self.epochs):
            error = 0.0
            for x, y in zip(X, Y):
                y_pred = np.sign(np.dot(x, self.w))
                if y_pred != y:
                    error += 1
                    self.w += self.lr * y * x.reshape(self.w.shape)
            print(f"Error: {error}")


dataset = np.array(
    [
        [2.7810836, 2.550537003, -1],
        [1.465489372, 2.362125076, -1],
        [3.396561688, 4.400293529, -1],
        [1.38807019, 1.850220317, -1],
        [3.06407232, 3.005305973, -1],
        [7.627531214, 2.759262235, 1],
        [5.332441248, 2.088626775, 1],
        [6.922596716, 1.77106367, 1],
        [8.675418651, -0.242068655, 1],
        [7.673756466, 3.508563011, 1],
    ]
)
X = dataset[:, :-1]
Y = dataset[:, -1]
l_rate = 0.1
n_epoch = 5
p = Perceptron(l_rate, n_epoch)
p.fit(X, Y)
