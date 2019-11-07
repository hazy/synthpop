import numpy as np
import pandas as pd

from methods import SampleMethod, CARTMethod, NormMethod, NormRankMethod, PolyregMethod
from preprocess import df, columns, dtypes, processing_numtocat_dict, processing_nan_dict, cont_na, numtocat, catgroups
from arguments import method, visit_sequence, proper, smoothing


# built under the hoood variables
seed = 13
n_columns = len(columns)
columns_processed = df.columns.tolist()
n_columns_processed = len(columns_processed)


# check or build method
empty_method = ''
sample_method = 'sample'
# non-parametric methods
cart_method = 'cart'
# parametric methods
parametric_method = 'parametric'
norm_method = 'norm'
normrank_method = 'normrank'
polyreg_method = 'polyreg'

all_methods = [empty_method, sample_method, cart_method, parametric_method, norm_method, normrank_method, polyreg_method]
na_methods = [cart_method, norm_method, normrank_method, polyreg_method]
default_method = cart_method

if method is None:
    method = [default_method for _ in columns]
elif isinstance(method, str):
    if method == cart_method:
        method = [cart_method for _ in columns]
    if method == parametric_method:
        method = [normrank_method if dtypes[col] in ['int', 'float', 'datetime'] else polyreg_method for col in columns]
else:
    assert all([m in all_methods for m in method])
    assert sample_method in method
assert len(method) == n_columns
method = pd.Series(method, index=columns)


# check or build visit_sequence
if visit_sequence is None:
    visit_sequence = np.arange(n_columns)
if isinstance(visit_sequence, np.ndarray):
    visit_sequence = [i.item() for i in visit_sequence]
assert all([isinstance(i, int) for i in visit_sequence]) or all([isinstance(col, str) for col in visit_sequence])
assert len(set(visit_sequence)) == len(visit_sequence)
if isinstance(visit_sequence[0], int):
    assert set(visit_sequence).issubset(set(np.arange(n_columns)))
    visit_sequence = [columns[i] for i in visit_sequence]
else:
    assert set(visit_sequence).issubset(set(columns))
visited_columns = [col for col in columns if col in visit_sequence]
visit_sequence = pd.Series([visit_sequence.index(col) for col in visited_columns], index=visited_columns)


# adjust the method for first visited colum in visit_sequence with sample_method
method[visit_sequence.index[visit_sequence == 0].tolist()] = sample_method


# define predictor_matrix
predictor_matrix = np.zeros([len(visit_sequence.index), len(visit_sequence.index)], dtype=int)
predictor_matrix = pd.DataFrame(predictor_matrix, index=visit_sequence.index, columns=visit_sequence.index)
visited_columns = []
for col, _ in visit_sequence.sort_values().iteritems():
    predictor_matrix.loc[col, visited_columns] = 1
    visited_columns.append(col)


# check proper
assert proper in [True, False]


# check smoothing
if smoothing is None:
    smoothing = {col: False for col in columns}
elif isinstance(smoothing, str):
    assert smoothing == 'density'
    smoothing = {col: dtypes[col] in ['int', 'float', 'datetime'] for col in columns}
else:
    assert all([(smoothing_method == 'density' and dtypes[col] in ['int', 'float', 'datetime']) or smoothing_method is None for col, smoothing_method in smoothing.items()])
    smoothing = {col: (smoothing.get(col, None) == 'density' and dtypes[col] in ['int', 'float', 'datetime']) for col in columns}


# adjust method, visit_sequence, predictor_matrix and smoothing according to cont_na and processing_nan_dict
cont_na_method_map = {cart_method: cart_method,
                      norm_method: polyreg_method,
                      normrank_method: polyreg_method,
                      polyreg_method: polyreg_method
                      }

for col in method.index:
    if col in cont_na and method[col] in na_methods:
        nan_col_index = method.index.get_loc(col)
        index_list = method.index.tolist()
        index_list.insert(nan_col_index, columns_processed[nan_col_index])
        method = method.reindex(index_list, fill_value=cont_na_method_map[method[col]])

for col in visit_sequence.index:
    if col in cont_na and method[col] in na_methods:
        visit_step = visit_sequence[col]
        visit_sequence.loc[visit_sequence >= visit_step] += 1

        nan_col_index = visit_sequence.index.get_loc(col)
        index_list = visit_sequence.index.tolist()
        index_list.insert(nan_col_index, columns_processed[nan_col_index])
        visit_sequence = visit_sequence.reindex(index_list, fill_value=visit_step)

for col in predictor_matrix:
    if col in cont_na and method[col] in na_methods:
        nan_col_index = predictor_matrix.columns.get_loc(col)
        predictor_matrix.insert(nan_col_index, columns_processed[nan_col_index], predictor_matrix[col])

        index_list = predictor_matrix.index.tolist()
        index_list.insert(nan_col_index, columns_processed[nan_col_index])
        predictor_matrix = predictor_matrix.reindex(index_list, fill_value=0)
        predictor_matrix.loc[columns_processed[nan_col_index]] = predictor_matrix.loc[col]

        predictor_matrix.loc[col, columns_processed[nan_col_index]] = 1

for col in cont_na:
    smoothing[processing_nan_dict[col]['col_nan_name']] = False


# adjust method, smoothing according to numtocat and processing_numtocat_dict
for col in numtocat:
    method[col] = cont_na_method_map[method[col]]

# # this is already taken care of
# for col in numtocat:
#     smoothing[col] = False


minibucket = 5

saved_models = {}


# train
predictor_matrix_columns = predictor_matrix.columns.to_numpy()
for col, visit_step in visit_sequence.sort_values().iteritems():
    print('train_{}'.format(col))

    # initialise the method
    if method[col] == sample_method:
        col_method = SampleMethod(dtypes[col], smoothing=smoothing[col], proper=proper, random_state=seed)

    elif method[col] == cart_method:
        col_method = CARTMethod(dtypes[col], smoothing=smoothing[col], proper=proper, minibucket=minibucket, random_state=seed)

    elif method[col] == norm_method:
        col_method = NormMethod(dtypes[col], smoothing=smoothing[col], proper=proper, random_state=seed)

    elif method[col] == normrank_method:
        col_method = NormRankMethod(dtypes[col], smoothing=smoothing[col], proper=proper, random_state=seed)

    elif method[col] == polyreg_method:
        col_method = PolyregMethod(dtypes[col], proper=proper, random_state=seed)

    else:
        raise('method not supported.. what is going on dude?')

    # extract the relevant data and fit the method
    if method[col] == sample_method:
        x = df[col]

        col_method.fit(x)

    elif method[col] in na_methods:
        col_predictors = predictor_matrix_columns[predictor_matrix.loc[col].to_numpy() != 0]
        X = df[col_predictors]
        y = df[col]

        # extract the not nan values and only use them for fitting the method
        if col in cont_na:
            not_nan_indices = df[processing_nan_dict[col]['col_nan_name']] == 0
            X = X.loc[not_nan_indices]
            y = y.loc[not_nan_indices]

        col_method.fit(X, y)

    # save the method
    saved_models[col] = col_method
