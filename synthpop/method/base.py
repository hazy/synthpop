import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class Method(ABC):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

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
