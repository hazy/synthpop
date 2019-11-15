import numpy as np
import pandas as pd

# global variables
from synthpop import NUM_COLS_DTYPES, CAT_COLS_DTYPES

NAN_DICT = 'nan_dict'
NUMTOCAT_DICT = 'numtocat_dict'


class Processor:
    def __init__(self, synthpop_instance):
        self.spop = synthpop_instance
        self.processing_dict = {NUMTOCAT_DICT: {},
                                NAN_DICT: {}
                                }

    def preprocess(self, df, dtypes):
        for col in self.spop.visited_columns:
            # transform numerical columns in numtocat to categorical
            if col in self.spop.numtocat:
                self.processing_dict[NUMTOCAT_DICT][col] = {'dtype': self.spop.df_dtypes[col],
                                                            'categories': {}
                                                            }
                # take care of the NaNs
                # col_nan_category = 'NaN_category'
                col_nan_category = self.spop.catgroups[col]
                if any(df[col].isna()) or col in self.spop.cont_na:
                    self.processing_dict[NUMTOCAT_DICT][col]['categories'][col_nan_category] = []
                    df[col] = df[col].astype(object)

                    if any(df[col].isna()):
                        nan_indices = df[col].isna()
                        self.processing_dict[NUMTOCAT_DICT][col]['categories'][col_nan_category].append(df.loc[nan_indices, col].to_numpy())
                        df.loc[nan_indices, col] = col_nan_category

                    if col in self.spop.cont_na:
                        for col_nan_value in self.spop.cont_na[col]:
                            nan_indices = df[col] == col_nan_value
                            self.processing_dict[NUMTOCAT_DICT][col]['categories'][col_nan_category].append(df.loc[nan_indices, col].to_numpy())
                            df.loc[nan_indices, col] = col_nan_category

                    self.processing_dict[NUMTOCAT_DICT][col]['categories'][col_nan_category] = np.squeeze(self.processing_dict[NUMTOCAT_DICT][col]['categories'][col_nan_category])

                not_nan_indices = df[col] != col_nan_category
                not_nan_values = df.loc[not_nan_indices, col].copy()
                df.loc[not_nan_indices, col], bins = pd.cut(df.loc[not_nan_indices, col], self.spop.catgroups[col],
                                                            labels=range(self.spop.catgroups[col]), retbins=True, include_lowest=True)

                for bin_label, (bin_min, bin_max) in enumerate(zip(bins, bins[1:])):
                    self.processing_dict[NUMTOCAT_DICT][col]['categories'][bin_label] = not_nan_values[(bin_min < not_nan_values) & (not_nan_values <= bin_max)].to_numpy()

                df[col] = df[col].astype('category')
                self.spop.df_dtypes[col] = 'category'

            # NaNs in category columns
            # need to process NaNs only as all other categories will be taken care automatically
            if self.spop.df_dtypes[col] in CAT_COLS_DTYPES:
                if any(df[col].isna()):
                    col_nan_category = 'NaN_category'
                    self.processing_dict[NAN_DICT][col] = {'dtype': self.spop.df_dtypes[col],
                                                           'nan_value': col_nan_category
                                                           }

                    df[col].cat.add_categories(col_nan_category, inplace=True)
                    df[col].fillna(col_nan_category, inplace=True)

            # NaNs in numerical columns
            if self.spop.df_dtypes[col] in NUM_COLS_DTYPES:
                if any(df[col].isna()) or col in self.spop.cont_na:
                    # insert new column in df
                    col_nan_name = col + '_NaN'
                    df.insert(df.columns.get_loc(col), col_nan_name, 0)

                    self.processing_dict[NAN_DICT][col] = {'col_nan_name': col_nan_name,
                                                           'dtype': self.spop.df_dtypes[col],
                                                           'nan_flags': {}
                                                           }
                    flag_count = 1

                    if any(df[col].isna()):
                        self.processing_dict[NAN_DICT][col]['nan_flags'][flag_count] = np.nan
                        df.loc[df[col].isna(), col_nan_name] = flag_count
                        df.loc[df[col].isna(), col] = 0
                        flag_count += 1

                    if col in self.spop.cont_na:
                        for col_nan_value in self.spop.cont_na[col]:
                            self.processing_dict[NAN_DICT][col]['nan_flags'][flag_count] = col_nan_value
                            df.loc[df[col] == col_nan_value, col_nan_name] = flag_count
                            df.loc[df[col] == col_nan_value, col] = 0
                            flag_count += 1

                    df[col_nan_name] = df[col_nan_name].astype('category')
                    self.spop.df_dtypes[col_nan_name] = 'category'

        return df

    def postprocess(self, synth_df):
        for col, processing_numtocat_col_dict in self.processing_dict[NUMTOCAT_DICT].items():
            synth_df[col] = synth_df[col].astype('object')

            for category, category_values in processing_numtocat_col_dict['categories'].items():
                category_indices = synth_df[col] == category
                synth_df.loc[category_indices, col] = np.random.choice(category_values, size=category_indices.sum(), replace=True)

            # cast dtype back to original (float for int column with NaNs)
            if processing_numtocat_col_dict['dtype'] == 'int' and self.spop.catgroups[col] in processing_numtocat_col_dict['categories']:
                synth_df[col] = synth_df[col].astype(float)
            else:
                synth_df[col] = synth_df[col].astype(processing_numtocat_col_dict['dtype'])
            self.spop.df_dtypes[col] = processing_numtocat_col_dict['dtype']

        for col, processing_nan_col_dict in self.processing_dict[NAN_DICT].items():
            # NaNs in category columns
            # need to postprocess NaNs only all other categories will be taken care automatically
            if processing_nan_col_dict['dtype'] in CAT_COLS_DTYPES:
                col_nan_value = processing_nan_col_dict['nan_value']
                synth_df[col] = synth_df[col].astype(object)
                synth_df.loc[synth_df[col] == col_nan_value, col] = np.nan
                synth_df[col] = synth_df[col].astype('category')

            # NaNs in numerical columns
            if processing_nan_col_dict['dtype'] in NUM_COLS_DTYPES:
                for col_nan_flag, col_nan_value in processing_nan_col_dict['nan_flags'].items():
                    nan_flag_indices = synth_df[processing_nan_col_dict['col_nan_name']] == col_nan_flag
                    synth_df.loc[nan_flag_indices, col] = col_nan_value
                synth_df.drop(columns=processing_nan_col_dict['col_nan_name'], inplace=True)

        return synth_df
