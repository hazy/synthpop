import pandas as pd

from train import df, columns, dtypes, cont_na, numtocat, catgroups, processing_numtocat_dict, processing_nan_dict, visit_sequence, method, sample_method, na_methods, saved_models, predictor_matrix_columns, predictor_matrix


n_rows = len(df)
synth_df = pd.DataFrame(columns=visit_sequence.index)
for col in synth_df.columns:
    synth_df[col] = pd.Series(dtype=dtypes[col])


for col, visit_step in visit_sequence.sort_values().iteritems():
    print('generate_{}'.format(col))

    # reload the method
    col_method = saved_models[col]

    # extract the relevant data and use the method to predict
    if method[col] == sample_method:
        synth_df[col] = col_method.predict(n_rows)

    elif method[col] in na_methods:
        col_predictors = predictor_matrix_columns[predictor_matrix.loc[col].to_numpy() != 0]
        X = df[col_predictors]

        synth_df[col] = col_method.predict(X)

        # change all missing values to 0
        if col in cont_na:
            nan_indices = synth_df[processing_nan_dict[col]['col_nan_name']] != 0
            synth_df.loc[nan_indices, col] = 0

    else:
        raise('method not supported.. what is going on dude?')

    synth_df[col] = synth_df[col].astype(dtypes[col])
