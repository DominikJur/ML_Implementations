from collections import Counter

import numpy as np

from algos.base import IEstimator
from algos.tree import DecisionTreeClassifier


class RandomForestClassifier(IEstimator):

    def __init__(self, 
        n_trees=100, 
        max_depth=100, 
        min_samples_split=2, 
        n_features=None , 
        random_state=None):

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_featues=self.n_features,
            )
            X_sample, y_sample = self._bootstrap_sample(X, y)

            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
        return self
    
    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        sample_indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[sample_indices, :], y[sample_indices]
    
    def predict(self, X):
        
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)

        y_pred = [Counter(tree_pred).most_common(1)[0][0] for tree_pred in tree_preds]
        return np.array(y_pred)