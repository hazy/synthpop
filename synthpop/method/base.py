import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from synthpop import NUM_COLS_DTYPES, CAT_COLS_DTYPES


class Method(ABC):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def prepare_dfs(self, X_df, y_df=None, normalise_num_cols=True, one_hot_cat_cols=True, fit=True):
        X_df = X_df.copy()

        if y_df is not None and self.dtype in NUM_COLS_DTYPES:
            y_df = y_df.copy()

            not_nan_indices = y_df.notna()
            X_df = X_df.loc[not_nan_indices]
            y_df = y_df.loc[not_nan_indices]

        if normalise_num_cols:
            if fit:
                num_cols = X_df.select_dtypes(NUM_COLS_DTYPES).columns.to_list()
                self.num_cols_range = {}
                for col in num_cols:
                    self.num_cols_range[col] = {'min': np.min(X_df[col]), 'max': np.max(X_df[col])}
                    X_df[col] = (X_df[col] - self.num_cols_range[col]['min']) / (self.num_cols_range[col]['max'] - self.num_cols_range[col]['min'])

            else:
                for col in self.num_cols_range:
                    X_df[col] = (X_df[col] - self.num_cols_range[col]['min']) / (self.num_cols_range[col]['max'] - self.num_cols_range[col]['min'])
                    X_df[col] = np.clip(X_df[col], 0, 1)

        if one_hot_cat_cols:
            # Avoid the Dummy Variable Trap
            # https://towardsdatascience.com/one-hot-encoding-multicollinearity-and-the-dummy-variable-trap-b5840be3c41a
            cat_cols = X_df.select_dtypes(CAT_COLS_DTYPES).columns.to_list()
            X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=True)

            if fit:
                self.train_cols = X_df.columns.tolist()

            else:
                test_cols = X_df.columns.tolist()
                missing_cols = set(self.train_cols) - set(test_cols)
                for col in missing_cols:
                    X_df[col] = 0

                X_df = X_df[self.train_cols]

        return X_df, y_df
