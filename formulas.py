import numpy as np


class Formulas:
    # BIAS/VARIANCE
    def bias(self, Y_pred, Y):
        # (E[Y_pred] - E[Y])^2
        return (np.mean(Y_pred) - np.mean(Y)) ** 2

    def variance(self, Y_pred):
        return np.mean(Y_pred - np.mean(Y_pred))

    def bias_variance_decomp(self, Y_pred, Y, irreducible_error):
        return self.bias(Y_pred, Y) ** 2 + self.variance(Y_pred) + irreducible_error

    # CONFUSION MATRIX
    def accuracy(self, tp, fp, fn, tn):
        return (tp + tn) / (tp + fp + fn + tn)

    def recall(self, tp, fp, fn, tn):
        return tp / (tp + fn)

    def precision(self, tp, fp, fn, tn):
        return tp / (tp + fp)

    def f1(self, tp, fp, fn, tn):
        return (2 * self.recall(tp, fp, fn, tn) * self.precision(tp, fp, fn, tn)) / (
            self.recall(tp, fp, fn, tn) + self.precision(tp, fp, fn, tn)
        )

    def type_1_error(self, tp, fp, fn, tn):
        return fp

    def type_2_error(self, tp, fp, fn, tn):
        return fn

    def true_pos_rate(self, tp, fp, fn, tn):
        # of the true ones, how many are positives
        return tp / (tp + fn)

    def false_pos_rate(self, tp, fp, fn, tn):
        # of the false ones, how many were incorrectly labeled as positive
        return fp / (fp + tn)

    # PENALTIES FOR REGULARIZATION
    def l2_penalty(self, w):
        return np.sum(np.square(w))

    def l1_penalty(self, w):
        return np.sum(np.abs(w))

    def elasticnet_penalty(self, w, alpha1, alpha2):
        # tradeoff between l1 and l2
        return alpha1 * self.l2_penalty(w) + alpha2 * self.l1_penalty(w1)

    # Basic losses
    def l2_loss(self, Y_pred, Y):
        return np.sum(np.square(Y_pred - Y))

    def l1_loss(self, Y_pred, Y):
        return np.sum(np.abs(Y_pred - Y))

    def pseudo_huber_loss(self, Y_pred, Y, alpha):
        error = self.l1_loss(Y_pred, Y)
        if error <= alpha:
            return self.l2_loss(Y_pred, Y)
        return error

    # REGRESSION LOSS FUNCTIONS
    # linear regression
    def mse(self, Y_pred, Y):
        return np.sum(np.square(Y_pred - Y)) / Y_pred.shape[0]

    def mae(self, Y_pred, Y):
        return np.sum(np.abs(Y_pred - Y)) / Y_pred.shape[0]

    # CLASSIFICATION LOSS FUNCTIONS
    # logistic regression
    def binary_cross_entropy(self, Y_pred, Y):
        # - 1/n sum (ylogpi + (1-y)log(1-pi))
        log_pi = np.log(Y_pred)
        log_1_minus_pi = np.log(1 - Y_pred)
        positive_class = np.multiply(Y, log_pi)
        negative_class = np.multiply(1 - Y, log_1_minus_pi)
        return -1 * (np.sum(positive_class + negative_class) / Y_pred.shape[0])

    def kl_divergence(self, p, q):
        def entropy(self, d):
            return np.sum(np.prod(d, np.log(d)))

        def cross_entropy(self, d1, d2):
            return np.sum(np.prod(d1, np.log(d2)))

        # non negative since entropy = minimum average lossless encoding size
        # cross entropy = if distributions are different, will have some additional loss
        # if P = Q, then KL divergence is 0

        return cross_entropy - entropy

    # SVM
    def hinge_loss(self, y, y_pred):
        # ensures that there is a margin of at least 1
        return max(0, 1 - y * y_pred)

    def svm_optimization(self, y, y_pred, w):
        return (1 / 2) * self.l2_penalty(w) + hinge_loss(y, y_pred)

    # PREPROCESSING
    def standardize(self, X):
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # ACTIVATION FUNCTIONS
    def sigmoid(self, a):
        return 1 / 1 + np.exp(-a)

    def softmax(self, y):
        exp = np.exp(y)
        return exp / np.sum(exp)

    def relu(self, a):
        return max(0, a)

    def leaky_relu(self, a, alpha):
        if a < 0:
            return alpha * a
        else:
            return a
