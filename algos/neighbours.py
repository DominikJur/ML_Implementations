from collections import Counter

import numpy as np

from algos.base import IEstimator


class KNN(IEstimator):
    def __init__(self, k_neighbors=3):
        self.k_neighbors = k_neighbors
    
    def validate(self, X, y=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(X) == 0:
            raise ValueError("Empty array X")
        if y is not None:
            if not isinstance(y, np.ndarray):
                y = np.array(y)
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            if len(y) == 0:
                raise ValueError("Empty array y")
        return X, y
    
    def fit(self, X, y):
        X, y = self.validate(X, y)
        self.X = X
        self.y = y

    def _euclidian_metric(self, x1, x2):
        assert len(x1) == len(x2)
        return np.sqrt(np.sum((x1-x2)**2))

    def _predict(self, x):
        distances = [self._euclidian_metric(x,x_) for x_ in self.X]

        k_nearest_ids = np.argsort(distances)[:self.k_neighbors]
        k_nearest_labels = [self.y[id][0] for id in k_nearest_ids]

        return Counter(k_nearest_labels).most_common()[0][0]


    def predict(self, X):
        X, _ = self.validate(X)
        return [self._predict(x) for x in X]