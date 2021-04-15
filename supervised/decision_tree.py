import numpy as np


class DecisionNode:
    def __init__(self, idx=None, value=None):
        self.idx = idx
        self.left = None
        self.right = None
        self.value = value


class DecisionTree:
    def __init__(self):
        self.root = None

    def gini_coef(self, X, Y):
        # calculate (1) the gini coef for class 0 (2) the gini coef for class 1
        # (1 - sum_{i=1}^n prop_i * prop_i) * (size of class)/(total size)
        possible_feat = set(X)
        total_label_gini = 0.0
        for f in possible_feat:
            curr_l_rows = Y[X == f]
            count_per_class = np.bincount(
                curr_l_rows
            )  # count_per_feat_val[i] will contain number of elts with that feature

            prop_per_class = count_per_class / len(curr_l_rows)
            sq_prop_per_feat_val = prop_per_class * prop_per_class

            feat_prior = curr_l_rows.shape[0] / X.shape[0]

            total_label_gini += (1 - np.sum(sq_prop_per_feat_val)) * feat_prior
        return total_label_gini

    def info_gain(self):
        pass

    def fit(self, X, Y):
        self.root = self._recursive_fit(X, Y, np.arange(X.shape[1]))

    def _recursive_fit(self, X, Y, possible_feats):
        if len(possible_feats) == 0 or len(set(Y)) == 1 or len(np.unique(X)) == 1:
            if len(set(Y)) > 1:
                bin_counts = np.bincount(Y)
                value = np.argmax(bin_counts)
            else:
                value = Y[0]
            d = DecisionNode(value=value)
            return d
        min_impurity_feature = None
        min_impurity = float("inf")
        for feat_idx in possible_feats:
            print(feat_idx)
            feat_col = X[:, feat_idx]
            if self.gini_coef(feat_col, Y) < min_impurity:
                min_impurity = self.gini_coef(feat_col, Y)
                min_impurity_feature = feat_idx
        n = DecisionNode(idx=min_impurity_feature)
        feature_list = np.arange(X.shape[1])

        feats_to_left = X[:, min_impurity_feature] == 0
        feats_to_right = X[:, min_impurity_feature] == 1
        if not np.any(feats_to_left) or np.any(feats_to_right) == 0:
            if len(set(Y)) > 1:
                bin_counts = np.bincount(Y)
                value = np.argmax(bin_counts)
            else:
                value = Y[0]
            d = DecisionNode(value=value)
            return d
        possible_feats = possible_feats[possible_feats != min_impurity_feature]
        n.left = self._recursive_fit(X[feats_to_left], Y[feats_to_left], possible_feats)
        n.right = self._recursive_fit(
            X[feats_to_right], Y[feats_to_right], possible_feats
        )
        return n

    def predict(self, X, Y):
        y_pred = []
        for x, y in zip(X, Y):
            curr_node = self.root
            while curr_node.value is None:
                idx = curr_node.idx
                if x[idx] == 0:
                    curr_node = curr_node.left
                else:
                    curr_node = curr_node.right
            y_pred.append((curr_node.value, y))
        return y_pred


dataset = np.array(
    [
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
)
X, Y = dataset[:, :-1], dataset[:, -1]
dt = DecisionTree()
dt.fit(X, Y)
print(dt.predict(X, Y))