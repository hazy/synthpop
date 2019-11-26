import numpy as np
import pandas as pd

# global variables
from synthpop import NUM_COLS_DTYPES, CAT_COLS_DTYPES

NAN_KEY = 'nan'
NUMTOCAT_KEY = 'numtocat'


class Processor:
    def __init__(self, spop):
        self.spop = spop
        self.processing_dict = {NUMTOCAT_KEY: {},
                                NAN_KEY: {}
                                }

    def preprocess(self, df, dtypes):
        for col in self.spop.visited_columns:
            col_nan_indices = df[col].isna()
            cont_nan_indices = {v: df[col] == v for v in self.spop.cont_na.get(col, [])}
            col_nan_series = [(np.nan, col_nan_indices)] + list(cont_nan_indices.items())

            col_all_nan_indices = pd.DataFrame({index: value[1] for index, value in enumerate(col_nan_series)}).max(axis=1)
            col_not_nan_indices = np.invert(col_all_nan_indices)

            # transform numerical columns in numtocat to categorical
            if col in self.spop.numtocat:
                self.processing_dict[NUMTOCAT_KEY][col] = {'dtype': self.spop.df_dtypes[col],
                                                           'categories': {}
                                                           }

                # Dealing With Non-NaN Values
                not_nan_values = df.loc[col_not_nan_indices, col].copy()
                df.loc[col_not_nan_indices, col] = pd.cut(df.loc[col_not_nan_indices, col], self.spop.catgroups[col], labels=range(self.spop.catgroups[col]), include_lowest=True)

                grouped = pd.DataFrame({'grouped': df.loc[col_not_nan_indices, col], 'real': not_nan_values}).groupby('grouped')
                self.processing_dict[NUMTOCAT_KEY][col]['categories'] = grouped['real'].apply(np.array).to_dict()

                # Dealing with NaN
                for index, (_, bool_series) in enumerate(col_nan_series):
                    nan_cat = self.spop.catgroups[col] + index
                    self.processing_dict[NUMTOCAT_KEY][col]['categories'][nan_cat] = df.loc[bool_series, col].to_numpy()
                    df.loc[bool_series, col] = nan_cat

                df[col] = df[col].astype('category')
                self.spop.df_dtypes[col] = 'category'

            else:
                # NaNs in category columns
                # need to process NaNs only as all other categories will be taken care automatically
                if self.spop.df_dtypes[col] in CAT_COLS_DTYPES:
                    if col_nan_indices.any():
                        # TODO beware of 'NaN_category' naming
                        col_nan_category = 'NaN_category'
                        self.processing_dict[NAN_KEY][col] = {'dtype': self.spop.df_dtypes[col],
                                                              'nan_value': col_nan_category
                                                              }

                        df[col].cat.add_categories(col_nan_category, inplace=True)
                        df[col].fillna(col_nan_category, inplace=True)

                # NaNs in numerical columns
                elif self.spop.df_dtypes[col] in NUM_COLS_DTYPES:
                    if col_all_nan_indices.any():
                        # insert new column in df
                        # TODO beware of '_NaN' naming
                        col_nan_name = col + '_NaN'
                        df.insert(df.columns.get_loc(col), col_nan_name, 0)

                        self.processing_dict[NAN_KEY][col] = {'col_nan_name': col_nan_name,
                                                              'dtype': self.spop.df_dtypes[col],
                                                              'nan_flags': {}
                                                              }

                        for index, (cat, bool_series) in enumerate(col_nan_series):
                            cat_index = index + 1
                            self.processing_dict[NAN_KEY][col]['nan_flags'][cat_index] = cat
                            df.loc[bool_series, col_nan_name] = cat_index
                        df.loc[col_all_nan_indices, col] = 0

                        df[col_nan_name] = df[col_nan_name].astype('category')
                        self.spop.df_dtypes[col_nan_name] = 'category'

        return df

    def postprocess(self, synth_df):
        for col, processing_numtocat_col_dict in self.processing_dict[NUMTOCAT_KEY].items():
            synth_df[col] = synth_df[col].astype(object)
            col_synth_df = synth_df[col].copy()

            for category, category_values in processing_numtocat_col_dict['categories'].items():
                category_indices = col_synth_df == category
                synth_df.loc[category_indices, col] = np.random.choice(category_values, size=category_indices.sum(), replace=True)

            # cast dtype back to original (float for int column with NaNs)
            if synth_df[col].isna().any() and processing_numtocat_col_dict['dtype'] == 'int':
                synth_df[col] = synth_df[col].astype(float)
            else:
                synth_df[col] = synth_df[col].astype(processing_numtocat_col_dict['dtype'])
            # self.spop.df_dtypes[col] = processing_numtocat_col_dict['dtype']

        for col, processing_nan_col_dict in self.processing_dict[NAN_KEY].items():
            # NaNs in category columns
            # need to postprocess NaNs only all other categories will be taken care automatically
            if processing_nan_col_dict['dtype'] in CAT_COLS_DTYPES:
                col_nan_value = processing_nan_col_dict['nan_value']
                synth_df[col] = synth_df[col].astype(object)
                synth_df.loc[synth_df[col] == col_nan_value, col] = np.nan
                synth_df[col] = synth_df[col].astype('category')

            # NaNs in numerical columns
            elif processing_nan_col_dict['dtype'] in NUM_COLS_DTYPES:
                for col_nan_flag, col_nan_value in processing_nan_col_dict['nan_flags'].items():
                    nan_flag_indices = synth_df[processing_nan_col_dict['col_nan_name']] == col_nan_flag
                    synth_df.loc[nan_flag_indices, col] = col_nan_value
                synth_df.drop(columns=processing_nan_col_dict['col_nan_name'], inplace=True)

        return synth_df
