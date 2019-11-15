import numpy as np

from synthpop.method import Method, proper, smooth
# global variables
from synthpop import NUM_COLS_DTYPES


class SampleMethod(Method):
    def __init__(self, dtype, smoothing=False, proper=False, random_state=None):
        self.dtype = dtype
        self.smoothing = smoothing
        self.proper = proper
        self.random_state = random_state

    def fit(self, x_df):
        if self.proper:
            x_df = proper(x_df)
        if self.dtype in NUM_COLS_DTYPES:
            self.x_real_min, self.x_real_max = np.min(x_df), np.max(x_df)

        self.values = x_df.to_numpy()

    def predict(self, n):
        y_pred = np.random.choice(self.values, size=n, replace=True)

        if self.smoothing and self.dtype in NUM_COLS_DTYPES:
            y_pred = smooth(self.dtype, y_pred, self.x_real_min, self.x_real_max)

        return y_pred
