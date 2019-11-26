import numpy as np
from numpy.linalg import inv, cholesky
from sklearn.linear_model import Ridge

from synthpop.method import Method, smooth
# global variables
from synthpop import NUM_COLS_DTYPES


class NormMethod(Method):
    def __init__(self, dtype, smoothing=False, proper=False, random_state=None, ridge=.00001, *args, **kwargs):
        self.dtype = dtype
        self.smoothing = smoothing
        self.proper = proper
        self.random_state = random_state
        self.alpha = ridge

        assert self.dtype in NUM_COLS_DTYPES
        self.norm = Ridge(alpha=self.alpha, random_state=self.random_state)

    def fit(self, X_df, y_df):
        X_df, y_df = self.prepare_dfs(X_df=X_df, y_df=y_df, normalise_num_cols=True, one_hot_cat_cols=True)
        self.y_real_min, self.y_real_max = np.min(y_df), np.max(y_df)
        n_rows, n_cols = X_df.shape

        X = X_df.to_numpy()
        y = y_df.to_numpy()
        self.norm.fit(X, y)

        residuals = y - self.norm.predict(X)

        if self.proper:
            # looks like proper is not working quite yet as it produces negative values for a strictly possitive column

            # Draws values of beta and sigma for Bayesian linear regression synthesis of y given x according to Rubin p.167
            # https://link.springer.com/article/10.1007/BF02924688
            self.sigma = np.sqrt(np.sum(residuals**2) / np.random.chisquare(n_rows - n_cols))
            # NOTE: I don't like the use of inv()
            V = inv(np.matmul(X.T, X))
            self.norm.coef_ += np.matmul(cholesky((V + V.T) / 2), np.random.normal(scale=self.sigma, size=n_cols))
        else:
            self.sigma = np.sqrt(np.sum(residuals**2) / (n_rows - n_cols - 1))

    def predict(self, X_test_df):
        X_test_df, _ = self.prepare_dfs(X_df=X_test_df, normalise_num_cols=True, one_hot_cat_cols=True, fit=False)
        n_test_rows = len(X_test_df)

        X_test = X_test_df.to_numpy()
        y_pred = self.norm.predict(X_test) + np.random.normal(scale=self.sigma, size=n_test_rows)

        if self.smoothing:
            y_pred = smooth(self.dtype, y_pred, self.y_real_min, self.y_real_max)

        return y_pred
