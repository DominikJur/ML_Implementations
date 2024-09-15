from abc import ABC, abstractmethod


class IEstimator(ABC):
    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError("NOT IMPLEMENTED")

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError("NOT IMPLEMENTED")
