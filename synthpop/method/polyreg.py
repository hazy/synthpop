import numpy as np
from sklearn.linear_model import LogisticRegression

from synthpop.method import Method, proper
# global variables
from synthpop import CAT_COLS_DTYPES


class PolyregMethod(Method):
    def __init__(self, dtype, proper=False, random_state=None, *args, **kwargs):
        self.dtype = dtype
        self.proper = proper
        self.random_state = random_state

        assert self.dtype in CAT_COLS_DTYPES
        # Specify solver and multi_class to silence this warning
        self.polyreg = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000, random_state=self.random_state)

    def fit(self, X_df, y_df):
        if self.proper:
            X_df, y_df = proper(X_df=X_df, y_df=y_df, random_state=self.random_state)

        X_df, y_df = self.prepare_dfs(X_df=X_df, y_df=y_df, normalise_num_cols=True, one_hot_cat_cols=True)

        X = X_df.to_numpy()
        y = y_df.to_numpy()
        self.polyreg.fit(X, y)

    def predict(self, X_test_df):
        X_test_df, _ = self.prepare_dfs(X_df=X_test_df, normalise_num_cols=True, one_hot_cat_cols=True, fit=False)
        n_test_rows = len(X_test_df)

        X_test = X_test_df.to_numpy()
        y_pred_proba = self.polyreg.predict_proba(X_test)

        uniform_noise = np.random.uniform(size=[n_test_rows, 1])
        indices = np.sum(uniform_noise > np.cumsum(y_pred_proba, axis=1), axis=1).astype(int)
        y_pred = self.polyreg.classes_[indices]

        return y_pred
