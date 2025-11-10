from collections import defaultdict

import numpy as np

from base import IEstimator



class NaiveBayes(IEstimator):
    
    def __init__(self):
        self.labels_idx = defaultdict(list)
        self._a_priori = {}

    def _get_label_idx(self, y):
        self.labels_idx = defaultdict(list)
        for idx, label in enumerate(y):
            self.labels_idx[label].append(idx)
        return self.labels_idx
    
    def _get_a_priori(self): # P(y)
        self._a_priori = {label: len(idx) for label, idx in self.labels_idx}
        total = sum(self._a_priori.values())
        self._a_priori = {label: n/total for label, n in self._a_priori}
        return self._a_priori
    
    def _get_likelihood(self, X, alpha=0):
        likelihood = {}
        for label, idx in self.label_idx.items():
            likelihood[label] = X[idx,:].sum(axis=0)+alpha
            likelihood[label] /= len(idx) + 2 * alpha

        return likelihood
    
    def _get_posteriori(self, X, alpha=0):
        pass
    
    
    
    
        