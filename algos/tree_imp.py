import numpy as np
from collections import Counter


def most_common_label(y): # Helper
    assert len(y) > 0
    counter = Counter(y)
    return counter.most_common(1)[0][0]


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


class TreeClassifier:
    def __init__(self, max_depth=10, criterion="gini"):
        assert criterion in ("gini", "entropy"), "Criterion must be 'gini' or 'entropy'"
        self.max_depth = max_depth
        self.criterion = criterion
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)
        return self

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        node = self.root
        while not node.is_leaf_node():
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _split(self, X_column, split_threshold):
        left_indices = np.argwhere(X_column <= split_threshold).flatten()
        right_indices = np.argwhere(X_column > split_threshold).flatten()
        return left_indices, right_indices

    def _grow_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            leaf_value = most_common_label(y)
            return Node(value=leaf_value)

        n_features = X.shape[1]
        feature_ids = np.arange(n_features)

        best_feature_, best_threshold_ = self._best_split(X, y, feature_ids)
        left_indices, right_indices = self._split(X[:, best_feature_], best_threshold_)
        
        left = self._grow_tree(X[left_indices, :], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices, :], y[right_indices], depth + 1)

        return Node(best_feature_, best_threshold_, left, right)

    def _best_split(self, X, y, feature_indices):
        split_index, split_threshold = None, None

        if self.criterion == "entropy":
            best_gain = -1

            for feature_index in feature_indices:
                X_column = X[:, feature_index]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
        
                    gain = self._information_gain(y, X_column, threshold)

                    if gain > best_gain:
                        best_gain = gain
                        split_index = feature_index
                        split_threshold = threshold
        else:  # gini
            best_gini = float("inf")

            for feature_index in feature_indices:
                X_column = X[:, feature_index]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    gini = self._weighted_gini(y, X_column, threshold)

                    if gini < best_gini:
                        best_gini = gini
                        split_index = feature_index
                        split_threshold = threshold

        return split_index, split_threshold

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        P = counts / len(y)
        return -np.sum([p * np.log2(p) for p in P if p > 0])

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

        return parent_entropy - child_entropy if n_l > 0 and n_r > 0 else 0

    def _gini(self, y):
        n = len(y)
        _, counts = np.unique(y, return_counts=True)
        P = counts / n
        gini = 1 - np.sum(P**2)
        return gini

    def _weighted_gini(self, y, X_column, split_threshold):
        left_indices, right_indices = self._split(X_column, split_threshold)
        n = len(y)
        n_l, n_r = len(left_indices), len(right_indices)

        if n == 0 or n_l == 0 or n_r == 0:
            return float("inf")

        if n_l == 0 or n_r == 0:
            return 0

        gini_left = self._gini(y[left_indices])
        gini_right = self._gini(y[right_indices])

        return (n_l / n) * gini_left + (n_r / n) * gini_right

class TreeRegressor:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        
    def fit(self, X, y):
        self.root = self._grow_tree(X, y)
        return self

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        node = self.root
        while not node.is_leaf_node():
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _split(self, X_column, split_threshold):
        left_indices = np.argwhere(X_column <= split_threshold).flatten()
        right_indices = np.argwhere(X_column > split_threshold).flatten()
        return left_indices, right_indices

    def _grow_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            leaf_value = most_common_label(y)
            return Node(value=leaf_value)

        n_features = X.shape[1]
        feature_ids = np.arange(n_features)

        best_feature_, best_threshold_ = self._best_split(X, y, feature_ids)
        left_indices, right_indices = self._split(X[:, best_feature_], best_threshold_)
        
        left = self._grow_tree(X[left_indices, :], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices, :], y[right_indices], depth + 1)

        return Node(best_feature_, best_threshold_, left, right)

    def _best_split(self, X, y, feature_indices):
        split_index, split_threshold = None, None

        best_ssr = float("inf")

        for feature_index in feature_indices:
            X_column = X[:, feature_index]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                ssr = self._sum_of_squared_residuals(y, X_column, threshold)

                if ssr < best_ssr:
                    best_ssr = ssr
                    split_index = feature_index
                    split_threshold = threshold

        return split_index, split_threshold
    
    def _sum_of_squared_residuals(self, y, X_column, split_threshold):
        left_indices, right_indices = self._split(X_column, split_threshold)
        
        if len(left_indices) == 0 or len(right_indices) == 0:
            return float("inf")

        left_y = y[left_indices]
        right_y = y[right_indices]

        left_mean = np.mean(left_y)
        right_mean = np.mean(right_y)

        ssr = np.sum((left_y - left_mean) ** 2) + np.sum((right_y - right_mean) ** 2)

        return ssr
    
    
    
    
    
class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=10, criterion="gini"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            tree = TreeClassifier(max_depth=self.max_depth, 
                                  criterion=self.criterion)
            tree.fit(X, y)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([most_common_label(tree_pred) for tree_pred in tree_preds.T])
    

class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=10):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            tree = TreeRegressor(max_depth=self.max_depth)
            tree.fit(X, y)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)
    
    
class SimpleGBR:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_pred = None
    
    def fit(self, X, y):
        self.initial_pred = np.mean(y)
        F = np.full(len(y), self.initial_pred) 
        
        for _ in range(self.n_estimators):
            residuals = y - F
            tree = TreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)
            F += self.learning_rate * tree.predict(X)
    
    def predict(self, X):
        y_pred = np.full(X.shape[0], self.initial_pred)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred