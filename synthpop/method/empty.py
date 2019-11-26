import numpy as np

from synthpop.method import Method


class EmptyMethod(Method):
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, X_df, y_df):
        pass

    def predict(self, X_test_df):
        n = X_test_df.shape[0]

        y_pred = np.empty(n) * np.nan

        return y_pred
