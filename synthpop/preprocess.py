import numpy as np
import pandas as pd

from load import df, columns, dtypes
from arguments import cont_na, numtocat, catgroups


# check cont_na
if cont_na is None:
    cont_na = {}
else:
    assert all([col in columns for col in cont_na])
    assert all([dtypes[col] in ['int', 'float', 'datetime'] for col in cont_na])


# check numtocat
if numtocat is None:
    numtocat = []
else:
    assert all([col in columns for col in numtocat])
    assert all([dtypes[col] in ['int', 'float', 'datetime'] for col in numtocat])
    # assert all([col not in cont_na for col in numtocat])


# check numtocat
if catgroups is None:
    catgroups = {col: 5 for col in numtocat}
elif isinstance(catgroups, int):
    assert catgroups > 1
    catgroups = {col: catgroups for col in numtocat}
else:
    assert set(list(catgroups.keys())) == set(numtocat)
    assert all([(isinstance(n_groups, int) and n_groups > 1) for n_groups in catgroups.values()])


processing_nan_dict = {}
processing_numtocat_dict = {}


for col in df:
    # transform numerical columns in numtocat to categorical
    if col in numtocat:
        processing_numtocat_dict[col] = {'dtype': dtypes[col],
                                         'categories': {}
                                         }

        # take care of the NaNs
        # col_nan_category = 'NaN_category'
        col_nan_category = catgroups[col]
        if any(df[col].isna()) or col in cont_na:
            processing_numtocat_dict[col]['categories'][col_nan_category] = []
            df[col] = df[col].astype('object')

            if any(df[col].isna()):
                nan_indices = df[col].isna()
                processing_numtocat_dict[col]['categories'][col_nan_category].append(df.loc[nan_indices, col].to_numpy())
                df.loc[nan_indices, col] = col_nan_category

            if col in cont_na:
                for col_nan_value in cont_na[col]:
                    nan_indices = df[col] == col_nan_value
                    processing_numtocat_dict[col]['categories'][col_nan_category].append(df.loc[nan_indices, col].to_numpy())
                    df.loc[nan_indices, col] = col_nan_category

            processing_numtocat_dict[col]['categories'][col_nan_category] = np.squeeze(processing_numtocat_dict[col]['categories'][col_nan_category])

        not_nan_indices = df[col] != col_nan_category
        not_nan_values = df.loc[not_nan_indices, col].copy()
        df.loc[not_nan_indices, col], bins = pd.cut(df.loc[not_nan_indices, col], catgroups[col], labels=range(catgroups[col]), retbins=True, include_lowest=True)

        for bin_label, (bin_min, bin_max) in enumerate(zip(bins, bins[1:])):
            processing_numtocat_dict[col]['categories'][bin_label] = not_nan_values[(bin_min < not_nan_values) & (not_nan_values <= bin_max)].to_numpy()

        df[col] = df[col].astype('category')
        dtypes[col] = 'category'

    # NaNs in category columns
    # need to process NaNs only as all other categories will be taken care automatically
    if dtypes[col] in ['category', 'bool']:
        if any(df[col].isna()):
            col_nan_category = 'NaN_category'
            processing_nan_dict[col] = {'dtype': dtypes[col],
                                        'nan_value': col_nan_category
                                        }

            df[col].cat.add_categories(col_nan_category, inplace=True)
            df[col].fillna(col_nan_category, inplace=True)

    # NaNs in numerical columns
    if dtypes[col] in ['int', 'float', 'datetime']:
        if any(df[col].isna()) or col in cont_na:
            # insert new column in df
            col_nan_name = col + '_NaN'
            df.insert(df.columns.get_loc(col), col_nan_name, 0)

            processing_nan_dict[col] = {'col_nan_name': col_nan_name,
                                        'dtype': dtypes[col],
                                        'nan_flags': {}
                                        }
            flag_count = 1

            if any(df[col].isna()):
                processing_nan_dict[col]['nan_flags'][flag_count] = np.nan
                df.loc[df[col].isna(), col_nan_name] = flag_count
                df.loc[df[col].isna(), col] = 0
                flag_count += 1

            if col in cont_na:
                for col_nan_value in cont_na[col]:
                    processing_nan_dict[col]['nan_flags'][flag_count] = col_nan_value
                    df.loc[df[col] == col_nan_value, col_nan_name] = flag_count
                    df.loc[df[col] == col_nan_value, col] = 0
                    flag_count += 1

            df[col_nan_name] = df[col_nan_name].astype('category')
            dtypes[col_nan_name] = 'category'
