import numpy as np
from scipy.stats import mode, iqr


def proper(X_df=None, y_df=None, random_state=None):
    sample_indicies = y_df.sample(frac=1, replace=True, random_state=random_state).index
    y_df = y_df.loc[sample_indicies]

    if X_df is None:
        return y_df

    else:
        X_df = X_df.loc[sample_indicies]
        return X_df, y_df


def smooth(dtype, y_synth, y_real_min, y_real_max):
    indices = [True for _ in range(len(y_synth))]

    # exclude from smoothing if freq for a single value higher than 70%
    y_synth_mode = mode(y_synth)
    if y_synth_mode.count / len(y_synth) > 0.7:
        indices = np.logical_and(indices, y_synth != y_synth_mode.mode)

    # exclude from smoothing if data are top-coded - approximate check
    y_synth_sorted = np.sort(y_synth)
    top_coded = 10 * np.abs(y_synth_sorted[-2]) < np.abs(y_synth_sorted[-1]) - np.abs(y_synth_sorted[-2])
    if top_coded:
        indices = np.logical_and(indices, y_synth != y_real_max)

    # R version
    # http://www.bagualu.net/wordpress/wp-content/uploads/2015/10/Modern_Applied_Statistics_With_S.pdf
    # R default (ned0) - [link eq5.5 in p127] - this is used as the other one is not a closed formula
    # R recommended (SJ) - [link eq5.6 in p129]
    bw = 0.9 * len(y_synth[indices]) ** -1/5 * np.minimum(np.std(y_synth[indices]), iqr(y_synth[indices]) / 1.34)

    # # Python version - much slower as it's not a closed formula and requires a girdsearch
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
