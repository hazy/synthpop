import numpy as np
from numpy.linalg import inv, cholesky
from scipy.stats import norm, rankdata

from synthpop.method import NormMethod, smooth


class NormRankMethod(NormMethod):
    # Adapted from norm by carrying out regression on Z scores from ranks
    # predicting new Z scores and then transforming back
    def fit(self, X_df, y_df):
        X_df, y_df = self.prepare_dfs(X_df=X_df, y_df=y_df, normalise_num_cols=True, one_hot_cat_cols=True)
        y_real_min, y_real_max = np.min(y_df), np.max(y_df)
        self.n_rows, n_cols = X_df.shape

        X = X_df.to_numpy()
        y = y_df.to_numpy()
        z = norm.ppf(rankdata(y).astype(int) / (self.n_rows + 1))
        self.norm.fit(X, z)

        residuals = z - self.norm.predict(X)

        if self.proper:
            # looks like proper is not working quite yet as it produces negative values for a strictly possitive column

            # Draws values of beta and sigma for Bayesian linear regression synthesis of y given x according to Rubin p.167
            # https://link.springer.com/article/10.1007/BF02924688
            self.sigma = np.sqrt(np.sum(residuals**2) / np.random.chisquare(self.n_rows - n_cols))
            # NOTE: I don't like the use of inv()
            V = inv(np.matmul(X.T, X))
            self.norm.coef_ += np.matmul(cholesky((V + V.T) / 2), np.random.normal(scale=self.sigma, size=n_cols))
        else:
            self.sigma = np.sqrt(np.sum(residuals**2) / (self.n_rows - n_cols - 1))

        if self.smoothing:
            y = smooth(self.dtype, y, y_real_min, y_real_max)

        self.y_sorted = np.sort(y)

    def predict(self, X_test_df):
        X_test_df, _ = self.prepare_dfs(X_df=X_test_df, normalise_num_cols=True, one_hot_cat_cols=True, fit=False)
        n_test_rows = len(X_test_df)

        X_test = X_test_df.to_numpy()
        z_pred = self.norm.predict(X_test) + np.random.normal(scale=self.sigma, size=n_test_rows)
        y_pred_indices = (norm.pdf(z_pred) * (self.n_rows + 1)).astype(int)
        y_pred_indices = np.clip(y_pred_indices, 1, self.n_rows)
        y_pred = self.y_sorted[y_pred_indices]

        return y_pred
