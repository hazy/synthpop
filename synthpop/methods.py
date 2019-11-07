import numpy as np
from numpy.linalg import inv, cholesky
from scipy.stats import norm, mode, iqr, rankdata
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import Ridge, LogisticRegression
# from sklearn.neighbors.kde import KernelDensity
# from sklearn.model_selection import GridSearchCV


class Method:
    def __init__(self):
        pass

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def prepare_X_df(self, X_df, normalise_num_cols=True, one_hot_cat_cols=True, fit=True):
        X_df = X_df.copy()

        if normalise_num_cols:
            if fit:
                num_cols = X_df.select_dtypes(['int', 'float', 'datetime']).columns.to_list()
                self.num_cols_range = {}
                for num_col in num_cols:
                    self.num_cols_range[num_col] = {'min': np.min(X_df[num_col]), 'max': np.max(X_df[num_col])}
                    X_df[num_col] = (X_df[num_col] - self.num_cols_range[num_col]['min']) / (self.num_cols_range[num_col]['max'] - self.num_cols_range[num_col]['min'])

            else:
                for num_col in self.num_cols_range:
                    X_df[num_col] = (X_df[num_col] - self.num_cols_range[num_col]['min']) / (self.num_cols_range[num_col]['max'] - self.num_cols_range[num_col]['min'])
                    X_df[num_col] = np.clip(X_df[num_col], 0, 1)

        if one_hot_cat_cols:
            # Avoid the Dummy Variable Trap
            # https://towardsdatascience.com/one-hot-encoding-multicollinearity-and-the-dummy-variable-trap-b5840be3c41a
            cat_cols = X_df.select_dtypes(['bool', 'category']).columns.to_list()
            X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=True)

            if fit:
                self.train_cols = X_df.columns.tolist()

            else:
                test_cols = X_df.columns.tolist()
                missing_cols = set(self.train_cols) - set(test_cols)
                for col in missing_cols:
                    X_df[col] = 0

                X_df = X_df[self.train_cols]

        return X_df


class SampleMethod(Method):
    def __init__(self, dtype, smoothing=False, proper=False, random_state=None):
        self.dtype = dtype
        self.smoothing = smoothing
        self.proper = proper
        self.random_state = random_state

    def fit(self, x_df):
        if self.proper:
            x_df = x_df.sample(frac=1, replace=True, random_state=self.random_state)
        if self.dtype in ['int', 'float', 'datetime']:
            self.x_real_min, self.x_real_max = np.min(x_df), np.max(x_df)

        self.values = x_df.to_numpy()

    def predict(self, n):
        y_pred = np.random.choice(self.values, size=n, replace=True)

        if self.smoothing and self.dtype in ['int', 'float', 'datetime']:
            y_pred = smooth(self.dtype, y_pred, self.x_real_min, self.x_real_max)

        return y_pred


class CARTMethod(Method):
    def __init__(self, dtype, smoothing=False, proper=False, minibucket=5, random_state=None):
        self.dtype = dtype
        self.smoothing = smoothing
        self.proper = proper
        self.minibucket = minibucket
        self.random_state = random_state

        if self.dtype in ['bool', 'category']:
            self.cart = DecisionTreeClassifier(min_samples_leaf=self.minibucket, random_state=self.random_state)
        elif self.dtype in ['int', 'float', 'datetime']:
            self.cart = DecisionTreeRegressor(min_samples_leaf=self.minibucket, random_state=self.random_state)
        else:
            raise('dtype not supported.. what is going on dude?')

    def fit(self, X_df, y_df):
        if self.proper:
            X_y_df = pd.concat([X_df, y_df], axis=1)
            X_y_df = X_y_df.sample(frac=1, replace=True, random_state=self.random_state)
            X_df = X_y_df.iloc[:, :-1]
            y_df = X_y_df.iloc[:, -1]

        X_df = self.prepare_X_df(X_df, normalise_num_cols=False, one_hot_cat_cols=True)
        if self.dtype in ['int', 'float', 'datetime']:
            self.y_real_min, self.y_real_max = np.min(y_df), np.max(y_df)

        X = X_df.to_numpy()
        y = y_df.to_numpy()
        self.cart.fit(X, y)

        # save the y distribution wrt trained tree nodes
        leaves = self.cart.apply(X)
        leaves_y_df = pd.DataFrame({'leaves': leaves, 'y': y})
        self.leaves_y_dict = leaves_y_df.groupby('leaves').apply(lambda x: x.to_numpy()[:, -1]).to_dict()

    def predict(self, X_test_df):
        X_test_df = self.prepare_X_df(X_test_df, normalise_num_cols=False, one_hot_cat_cols=True, fit=False)

        # predict the leaves and for each leaf randomly sample from the observed values
        X_test = X_test_df.to_numpy()
        leaves_pred = self.cart.apply(X_test)
        y_pred = np.zeros(len(leaves_pred), dtype=object)

        leaves_pred_index_df = pd.DataFrame({'leaves_pred': leaves_pred, 'index': range(len(leaves_pred))})
        leaves_pred_index_dict = leaves_pred_index_df.groupby('leaves_pred').apply(lambda x: x.to_numpy()[:, -1]).to_dict()
        for leaf, indices in leaves_pred_index_dict.items():
            y_pred[indices] = np.random.choice(self.leaves_y_dict[leaf], size=len(indices), replace=True)

        if self.smoothing and self.dtype in ['int', 'float', 'datetime']:
            y_pred = smooth(self.dtype, y_pred, self.y_real_min, self.y_real_max)

        return y_pred


class NormMethod(Method):
    def __init__(self, dtype, smoothing=False, proper=False, random_state=None, ridge=.00001):
        self.dtype = dtype
        self.smoothing = smoothing
        self.proper = proper
        self.random_state = random_state
        self.alpha = ridge

        assert self.dtype in ['int', 'float', 'datetime']
        self.norm = Ridge(alpha=self.alpha, random_state=self.random_state)

    def fit(self, X_df, y_df):
        X_df = self.prepare_X_df(X_df, normalise_num_cols=True, one_hot_cat_cols=True)
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
        X_test_df = self.prepare_X_df(X_test_df, normalise_num_cols=True, one_hot_cat_cols=True, fit=False)
        n_test_rows = len(X_test_df)

        X_test = X_test_df.to_numpy()
        y_pred = self.norm.predict(X_test) + np.random.normal(scale=self.sigma, size=n_test_rows)

        if self.smoothing:
            y_pred = smooth(self.dtype, y_pred, self.y_real_min, self.y_real_max)

        return y_pred


class NormRankMethod(NormMethod):
    # Adapted from norm by carrying out regression on Z scores from ranks
    # predicting new Z scores and then transforming back
    def fit(self, X_df, y_df):
        X_df = self.prepare_X_df(X_df, normalise_num_cols=True, one_hot_cat_cols=True)
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
        X_test_df = self.prepare_X_df(X_test_df, normalise_num_cols=True, one_hot_cat_cols=True, fit=False)
        n_test_rows = len(X_test_df)

        X_test = X_test_df.to_numpy()
        z_pred = self.norm.predict(X_test) + np.random.normal(scale=self.sigma, size=n_test_rows)
        y_pred_indices = (norm.pdf(z_pred) * (self.n_rows + 1)).astype(int)
        y_pred_indices = np.clip(y_pred_indices, 1, self.n_rows)
        y_pred = self.y_sorted[y_pred_indices]

        return y_pred


class PolyregMethod(Method):
    def __init__(self, dtype, proper=False, random_state=None):
        self.dtype = dtype
        self.proper = proper
        self.random_state = random_state

        assert self.dtype in ['bool', 'category']
        # Specify solver and multi_class to silence this warning
        self.polyreg = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000, random_state=self.random_state)

    def fit(self, X_df, y_df):
        if self.proper:
            X_y_df = pd.concat([X_df, y_df], axis=1)
            X_y_df = X_y_df.sample(frac=1, replace=True, random_state=self.random_state)
            X_df = X_y_df.iloc[:, :-1]
            y_df = X_y_df.iloc[:, -1]

        X_df = self.prepare_X_df(X_df, normalise_num_cols=True, one_hot_cat_cols=True)

        X = X_df.to_numpy()
        y = y_df.to_numpy()
        self.polyreg.fit(X, y)

    def predict(self, X_test_df):
        X_test_df = self.prepare_X_df(X_test_df, normalise_num_cols=True, one_hot_cat_cols=True, fit=False)
        n_test_rows = len(X_test_df)

        X_test = X_test_df.to_numpy()
        y_pred_proba = self.polyreg.predict_proba(X_test)

        uniform_noise = np.random.uniform(size=[n_test_rows, 1])
        indices = np.sum(uniform_noise > np.cumsum(y_pred_proba, axis=1), axis=1).astype(int)
        y_pred = self.polyreg.classes_[indices]

        return y_pred


class LogregMethod(PolyregMethod):
    pass


def smooth(dtype, y_synth, y_real_min, y_real_max):
    indices = [True for _ in range(len(y_synth))]

    # exclude from smoothing if freq for a single value higher than 70%
    y_synth_mode = mode(y_synth)
    if y_synth_mode.count / len(y_synth) > 0.7:
        indices = np.logical_and(indices, y_synth != y_synth_mode.mode)

    # exclude from smoothing if data are top-coded - approximate check
    top_coded = False
    y_synth_sorted = np.sort(y_synth)
    if 10 * np.abs(y_synth_sorted[-2]) < np.abs(y_synth_sorted[-1]) - np.abs(y_synth_sorted[-2]):
        top_coded = True
        indices = np.logical_and(indices, y_synth != y_real_max)

    # R version
    # http://www.bagualu.net/wordpress/wp-content/uploads/2015/10/Modern_Applied_Statistics_With_S.pdf
    # R default (ned0) - [link eq5.5 in p127] - this is used as the other one is not a closed formula
    # R recommended (SJ) - [link eq5.6 in p129]
    bw = 0.9 * len(y_synth[indices]) ** -1/5 * np.minimum(np.std(y_synth[indices]), iqr(y_synth[indices]) / 1.34)

    # # Python version - much slower as it's not a closed formula and requires a girdsearch
    # # TODO: use HazyOptimiser to find the optimal bandwidth
    # bandwidths = 10 ** np.linspace(-1, 1, 10)
    # grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bandwidths}, cv=3, iid=False)
    # grid.fit(y_synth[indices, None])
    # bw = grid.best_estimator_.bandwidth

    y_synth[indices] = np.array([np.random.normal(loc=value, scale=bw) for value in y_synth[indices]])
    if not top_coded:
        y_real_max += bw
    y_synth[indices] = np.clip(y_synth[indices], y_real_min, y_real_max)
    if dtype == 'int':
        y_synth[indices] = y_synth[indices].astype(int)

    return y_synth
