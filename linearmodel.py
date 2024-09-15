import numpy as np

from base import IEstimator


class LinReg(IEstimator):

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.theta = None
        self.fitted = False

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

    def fit(self, X, y, eta=0.1, n_iter=1000):
        X, y = self.validate(X, y)
        m = len(y)
        X_b = np.c_[np.ones((len(X), 1)), X]

        self.theta = np.zeros((len(X_b.T), 1))

        for _ in range(n_iter):

            nambla_mse = (2 / m) * (X_b.T @ (self._hypothesis(X_b) - y))
            self.theta -= eta * nambla_mse

        self.coef_ = self.theta[1:]
        self.intercept_ = self.theta[:1]

        self.fitted = True

        return self

    def _hypothesis(self, X):
        return X.dot(self.theta)

    def predict(self, X):
        X, _ = self.validate(X)
        if not self.fitted:
            raise ValueError("Model not fitted")
        return self._hypothesis(np.c_[np.ones((len(X), 1)), X])