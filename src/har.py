"""HAR model."""
from sklearn.linear_model import LinearRegression
import numpy as np


class HAR(LinearRegression):
    """
    HAR model.

    This is essentially a linear regression using HAR features.
    """

    def __init__(self):
        super().__init__()

    def fit(self, X, y, sample_weight=None):
        super().fit(X=X, y=y)
        return self

    def predict(self, X):
        return super().predict(X=X)
