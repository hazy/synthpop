import numpy as np

from generate import df, synth_df, dtypes, cont_na, numtocat, catgroups, processing_numtocat_dict, processing_nan_dict


for col, processing_numtocat_col_dict in processing_numtocat_dict.items():
    synth_df[col] = synth_df[col].astype('object')

    for category, category_values in processing_numtocat_col_dict['categories'].items():
        category_indices = synth_df[col] == category
        synth_df.loc[category_indices, col] = np.random.choice(category_values, size=category_indices.sum(), replace=True)

    # cast dtype back to original (float for int column with NaNs)
    if processing_numtocat_col_dict['dtype'] == 'int' and catgroups[col] in processing_numtocat_col_dict['categories']:
        synth_df[col] = synth_df[col].astype(float)
    else:
        synth_df[col] = synth_df[col].astype(processing_numtocat_col_dict['dtype'])
    dtypes[col] = processing_numtocat_col_dict['dtype']


for col, processing_nan_col_dict in processing_nan_dict.items():
    # NaNs in category columns
    # need to postprocess NaNs only all other categories will be taken care automatically
    if processing_nan_col_dict['dtype'] in ['category', 'bool']:
        col_nan_value = processing_nan_col_dict['nan_value']
        synth_df[col] = synth_df[col].astype(object)
        synth_df.loc[synth_df[col] == col_nan_value, col] = np.nan
        synth_df[col] = synth_df[col].astype('category')

    # NaNs in numerical columns
    if processing_nan_col_dict['dtype'] in ['int', 'float', 'datetime']:
        for col_nan_flag, col_nan_value in processing_nan_col_dict['nan_flags'].items():
            nan_flag_indices = synth_df[processing_nan_col_dict['col_nan_name']] == col_nan_flag
            synth_df.loc[nan_flag_indices, col] = col_nan_value
        synth_df.drop(columns=processing_nan_col_dict['col_nan_name'], inplace=True)


print(synth_df.head(100))
