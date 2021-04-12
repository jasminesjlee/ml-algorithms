import numpy as np
import pandas as pd
from collections import defaultdict


class NaiveBayes:
    def get_gaussian_prob(self, x, mean, std):
        exp_component = np.exp(-(1 / 2) * ((x - mean) / std) ** 2)
        return (1 / (std * np.sqrt(2 * np.pi))) * exp_component

    def separate_by_class(self, dataset, labels):
        class_to_stats = defaultdict()
        classes = set(labels)
        for c in classes:
            class_to_stats[c] = self.get_statistics(dataset[labels == c])
        return class_to_stats

    def get_statistics(self, dataset):
        return np.mean(dataset, axis=0), np.std(dataset, axis=0), len(dataset)

    def get_prior(self, class_size, total_size):
        return class_size / total_size

    def train(self, dataset):
        data = dataset[:, :-1]
        labels = dataset[:, -1]
        self.class_to_stats = self.separate_by_class(data, labels)
        self.total_size = len(dataset)

    def predict(self, test):
        y_pred = []
        data = test[:, :-1]
        labels = test[:, -1]
        for row, l in zip(data, labels):
            max_prob = float("-inf")
            max_class = None
            for c in self.class_to_stats:
                m, std, class_size = self.class_to_stats[c]
                curr_prob = self.get_prior(class_size, self.total_size) * np.prod(
                    self.get_gaussian_prob(row, m, std)
                )
                if curr_prob > max_prob:
                    max_prob = curr_prob
                    max_class = c
            y_pred.append((max_class, l))
        return y_pred


dataset = np.array(
    [
        [3.393533211, 2.331273381, 0],
        [3.110073483, 1.781539638, 0],
        [1.343808831, 3.368360954, 0],
        [3.582294042, 4.67917911, 0],
        [2.280362439, 2.866990263, 0],
        [7.423436942, 4.696522875, 1],
        [5.745051997, 3.533989803, 1],
        [9.172168622, 2.511101045, 1],
        [7.792783481, 3.424088941, 1],
        [7.939820817, 0.791637231, 1],
    ]
)

nv = NaiveBayes()
nv.train(dataset)
print(nv.predict(dataset))