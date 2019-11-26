import numpy as np
import pandas as pd

# classes
from synthpop.validator import Validator
from synthpop.processor import Processor
# global variables
from synthpop import NUM_COLS_DTYPES
from synthpop.processor import NAN_KEY
from synthpop.method import CART_METHOD, METHODS_MAP, NA_METHODS


class Synthpop:
    def __init__(self,
                 method=None,
                 visit_sequence=None,
                 # predictor_matrix=None,
                 proper=False,
                 cont_na=None,
                 smoothing=False,
                 default_method=CART_METHOD,
                 numtocat=None,
                 catgroups=None,
                 seed=None):
        # initialise the validator and processor
        self.validator = Validator(self)
        self.processor = Processor(self)

        # initialise arguments
        self.method = method
        self.visit_sequence = visit_sequence
        self.predictor_matrix = None
        self.proper = proper
        self.cont_na = cont_na
        self.smoothing = smoothing
        self.default_method = default_method
        self.numtocat = numtocat
        self.catgroups = catgroups
        self.seed = seed

        # check init
        self.validator.check_init()

    def fit(self, df, dtypes=None):
        # TODO check df and check/EXTRACT dtypes
        # - all column names of df are unique
        # - all columns data of df are consistent
        # - all dtypes of df are correct ('int', 'float', 'datetime', 'category', 'bool'; no object)
        # - can map dtypes (if given) correctly to df
        # should create map col: dtype (self.df_dtypes)

        self.df_columns = df.columns.tolist()
        self.n_df_rows, self.n_df_columns = np.shape(df)
        self.df_dtypes = dtypes

        # check processor
        self.validator.check_processor()
        # preprocess
        processed_df = self.processor.preprocess(df, self.df_dtypes)
        self.processed_df_columns = processed_df.columns.tolist()
        self.n_processed_df_columns = len(self.processed_df_columns)

        # check fit
        self.validator.check_fit()
        # fit
        self._fit(processed_df)

    def _fit(self, df):
        self.saved_methods = {}

        # train
        self.predictor_matrix_columns = self.predictor_matrix.columns.to_numpy()
        for col, visit_step in self.visit_sequence.sort_values().iteritems():
            print('train_{}'.format(col))

            # initialise the method
            col_method = METHODS_MAP[self.method[col]](dtype=self.df_dtypes[col], smoothing=self.smoothing[col], proper=self.proper, random_state=self.seed)
            # fit the method
            col_predictors = self.predictor_matrix_columns[self.predictor_matrix.loc[col].to_numpy() == 1]
            col_method.fit(X_df=df[col_predictors], y_df=df[col])
            # save the method
            self.saved_methods[col] = col_method

    def generate(self, k=None):
        self.k = k

        # check generate
        self.validator.check_generate()
        # generate
        synth_df = self._generate()
        # postprocess
        processed_synth_df = self.processor.postprocess(synth_df)

        return processed_synth_df

    def _generate(self):
        synth_df = pd.DataFrame(data=np.zeros([self.k, len(self.visit_sequence)]), columns=self.visit_sequence.index)

        for col, visit_step in self.visit_sequence.sort_values().iteritems():
            print('generate_{}'.format(col))

            # reload the method
            col_method = self.saved_methods[col]
            # predict with the method
            col_predictors = self.predictor_matrix_columns[self.predictor_matrix.loc[col].to_numpy() == 1]
            synth_df[col] = col_method.predict(synth_df[col_predictors])

            # change all missing values to 0
            if col in self.processor.processing_dict[NAN_KEY] and self.df_dtypes[col] in NUM_COLS_DTYPES and self.method[col] in NA_METHODS:
                nan_indices = synth_df[self.processor.processing_dict[NAN_KEY][col]['col_nan_name']] != 0
                synth_df.loc[nan_indices, col] = 0

            # map dtype to original dtype (only excpetion if column is full of NaNs)
            if synth_df[col].notna().any():
                synth_df[col] = synth_df[col].astype(self.df_dtypes[col])

        return synth_df
