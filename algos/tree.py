from collections import Counter

import numpy as np

from algos.base import IEstimator


class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeClassifier(IEstimator):
    def __init__(
        self, max_depth=100, min_samples_split=2, n_featues=None
    ):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_featues
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)
        
        return self

    def _grow_tree(self, X, y, depth=0):
        assert len(y) > 0

        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):


            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feature_ids = np.random.choice(n_features, self.n_features, replace=False)

        best_feature_, best_threshold_ = self._best_split(X, y, feature_ids)

        left_indices, right_indices = self._split(X[:, best_feature_], best_threshold_)

        left = self._grow_tree(X[left_indices, :], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices, :], y[right_indices], depth + 1)

        return Node(best_feature_, best_threshold_, left, right)

    def _best_split(self, X, y, feature_indices):
        best_gain = -1
        split_index, split_threshold = None, None
        
        for feature_index in feature_indices:
            X_column = X[:, feature_index]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_index = feature_index
                    split_threshold = threshold

        return split_index, split_threshold

    def _information_gain(self, y, X_column, split_threshold):
        parent_entropy = self._entropy(y)

        left_indices, right_indices = self._split(X_column, split_threshold)

        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_indices), len(right_indices)
        entropy_left = self._entropy(y[left_indices])
        entropy_right = self._entropy(y[right_indices])
        child_entropy = (n_l / n) * entropy_left + (n_r / n) * entropy_right

        return parent_entropy - child_entropy

    def _split(self, X_column, split_threshold):
        left_indices = np.argwhere(X_column <= split_threshold).flatten()
        right_indices = np.argwhere(X_column > split_threshold).flatten()
        return left_indices, right_indices

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        P = counts / len(y)
        return -np.sum([p * np.log2(p) for p in P if p > 0])

    def _most_common_label(self, y):
        assert len(y) > 0
        counter = Counter(y)
    
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
